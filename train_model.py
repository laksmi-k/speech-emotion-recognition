
import json
from pathlib import Path

import librosa
import numpy as np
import tensorflow as tf

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# -----------------------------
# 1. Configuration
# -----------------------------
DATA_DIR = Path(".")
SAMPLE_RATE = 22050
N_MFCC = 40
MAX_FRAMES = 180
EPOCHS = 40
BATCH_SIZE = 16

# RAVDESS emotion codes:
# 01 = neutral, 03 = happy, 04 = sad,
# 05 = angry, 06 = fearful
EMOTION_CODES = {
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "01": "neutral",
}

LABELS = ["angry", "fearful", "happy", "neutral", "sad"]
LABEL_TO_ID = {name: i for i, name in enumerate(LABELS)}


# -----------------------------
# 2. Extract MFCC features
# -----------------------------
def extract_features(audio_path):
    y, sr = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    if len(y) == 0:
        raise ValueError("Empty audio")

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=N_MFCC,
        n_fft=2048,
        hop_length=512
    )

    # Shape becomes (time frames, MFCC coefficients)
    mfcc = mfcc.T

    # Make all recordings the same length
    if mfcc.shape[0] > MAX_FRAMES:
        mfcc = mfcc[:MAX_FRAMES]
    elif mfcc.shape[0] < MAX_FRAMES:
        padding = MAX_FRAMES - mfcc.shape[0]
        mfcc = np.pad(
            mfcc,
            ((0, padding), (0, 0)),
            mode="constant"
        )

    return mfcc.astype(np.float32)


# -----------------------------
# 3. Load labeled recordings
# -----------------------------
def load_dataset():
    features = []
    labels = []
    actors = []

    audio_files = sorted(DATA_DIR.rglob("*.wav"))

    # Ignore virtual-environment files if present
    audio_files = [
        p for p in audio_files
        if ".venv" not in p.parts
    ]

    print("WAV files found:", len(audio_files))

    for path in audio_files:
        # RAVDESS filename example:
        # 03-01-05-01-02-01-12.wav
        parts = path.stem.split("-")

        if len(parts) != 7:
            continue

        emotion_code = parts[2]
        actor_id = parts[6]

        if emotion_code not in EMOTION_CODES:
            continue

        emotion = EMOTION_CODES[emotion_code]

        try:
            mfcc = extract_features(str(path))
            features.append(mfcc)
            labels.append(LABEL_TO_ID[emotion])
            actors.append(int(actor_id))
        except Exception as error:
            print("Skipping:", path, error)

    if not features:
        raise RuntimeError(
            "No usable RAVDESS files found. "
            "Check the dataset folder and filenames."
        )

    X = np.array(features, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)
    groups = np.array(actors)

    print("Usable recordings:", len(X))

    for index, name in enumerate(LABELS):
        print(name, ":", int(np.sum(y == index)))

    if len(np.unique(y)) != len(LABELS):
        raise RuntimeError(
            "Some emotion classes are missing. "
            "Check that the complete speech dataset was extracted."
        )

    return X, y, groups


# -----------------------------
# 4. Split by actor
# -----------------------------
def split_dataset(X, y, groups):
    # Keep test actors separate from training actors
    first_split = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42
    )

    train_val_idx, test_idx = next(
        first_split.split(X, y, groups)
    )

    # Split training actors into train and validation groups
    second_split = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=43
    )

    train_rel, val_rel = next(
        second_split.split(
            X[train_val_idx],
            y[train_val_idx],
            groups[train_val_idx]
        )
    )

    train_idx = train_val_idx[train_rel]
    val_idx = train_val_idx[val_rel]

    return train_idx, val_idx, test_idx


# -----------------------------
# 5. Build CNN-LSTM model
# -----------------------------
def build_model():
    inputs = tf.keras.Input(
        shape=(MAX_FRAMES, N_MFCC)
    )

    x = tf.keras.layers.Conv1D(
        64,
        kernel_size=5,
        padding="same",
        activation="relu"
    )(inputs)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)

    x = tf.keras.layers.Conv1D(
        128,
        kernel_size=3,
        padding="same",
        activation="relu"
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)

    x = tf.keras.layers.LSTM(64)(x)
    x = tf.keras.layers.Dropout(0.35)(x)

    outputs = tf.keras.layers.Dense(
        len(LABELS),
        activation="softmax"
    )(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# -----------------------------
# 6. Train and evaluate
# -----------------------------
def main():
    X, y, groups = load_dataset()

    train_idx, val_idx, test_idx = split_dataset(
        X, y, groups
    )

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Normalize using training data only
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True)
    std = np.maximum(std, 1e-6)

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    np.savez(
        "feature_stats.npz",
        mean=mean,
        std=std
    )

    with open("labels.json", "w") as file:
        json.dump(LABELS, file)

    model = build_model()
    model.summary()

    classes = np.unique(y_train)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train
    )

    class_weights = {
        int(class_id): float(weight)
        for class_id, weight in zip(classes, weights)
    }

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=7,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            "emotion_model.keras",
            monitor="val_loss",
            save_best_only=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            patience=3,
            factor=0.5
        )
    ]

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate only on previously held-out actors
    best_model = tf.keras.models.load_model(
        "emotion_model.keras"
    )

    loss, accuracy = best_model.evaluate(
        X_test,
        y_test,
        verbose=0
    )

    print("\nTest loss:", round(float(loss), 4))
    print("Test accuracy:", round(float(accuracy), 4))

    predictions = best_model.predict(
        X_test,
        verbose=0
    ).argmax(axis=1)

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=list(range(len(LABELS))),
            target_names=LABELS,
            zero_division=0
        )
    )

    print("Confusion matrix:")
    print(
        confusion_matrix(
            y_test,
            predictions,
            labels=list(range(len(LABELS)))
        )
    )

    print("\nTraining finished.")
    print("Saved model: emotion_model.keras")
    print("Saved labels: labels.json")
    print("Saved normalization values: feature_stats.npz")


if __name__ == "__main__":
    main()


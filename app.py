
import json
import os
import tempfile

import librosa
import numpy as np
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
import librosa.display

st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎙️",
    layout="wide"
)

# Model configuration must match train_model.py
SAMPLE_RATE = 22050
N_MFCC = 40
MAX_FRAMES = 180

MODEL_PATH = "emotion_model.keras"
LABELS_PATH = "labels.json"
STATS_PATH = "feature_stats.npz"


@st.cache_resource
def load_model_files():
    model = tf.keras.models.load_model(MODEL_PATH)

    with open(LABELS_PATH, "r") as f:
        labels = json.load(f)

    stats = np.load(STATS_PATH)
    mean = stats["mean"]
    std = np.maximum(stats["std"], 1e-6)

    return model, labels, mean, std


def extract_features(audio_path):
    y, sr = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    if len(y) == 0:
        raise ValueError("The audio file is empty.")

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=N_MFCC,
        n_fft=2048,
        hop_length=512
    ).T

    if mfcc.shape[0] > MAX_FRAMES:
        mfcc = mfcc[:MAX_FRAMES]
    elif mfcc.shape[0] < MAX_FRAMES:
        mfcc = np.pad(
            mfcc,
            ((0, MAX_FRAMES - mfcc.shape[0]), (0, 0)),
            mode="constant"
        )

    return y, sr, mfcc.astype(np.float32)


st.title("🎙️ Speech Emotion Recognition")
st.write(
    "CNN-LSTM-based speech emotion classification using MFCC features."
)

st.success("Using the trained CNN-LSTM model.")

try:
    model, labels, mean, std = load_model_files()
except Exception as error:
    st.error(
        "Could not load the trained model or supporting files. "
        "Check that emotion_model.keras, labels.json, and "
        "feature_stats.npz are in the same folder as app.py."
    )
    st.exception(error)
    st.stop()

audio_file = st.file_uploader(
    "Upload a speech recording",
    type=["wav", "mp3", "ogg", "flac", "m4a"]
)

if audio_file is not None:
    st.audio(audio_file)

    extension = os.path.splitext(audio_file.name)[1] or ".wav"
    audio_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=extension,
            delete=False
        ) as temp_file:
            temp_file.write(audio_file.getvalue())
            audio_path = temp_file.name

        y, sr, mfcc = extract_features(audio_path)

        st.subheader("Audio information")
        st.write(f"Sample rate: {sr} Hz")
        st.write(f"Duration: {len(y) / sr:.2f} seconds")

        st.subheader("MFCC features")
        fig, ax = plt.subplots()
        librosa.display.specshow(
            mfcc.T,
            x_axis="time",
            sr=sr,
            hop_length=512,
            ax=ax
        )
        ax.set_ylabel("MFCC coefficient")
        st.pyplot(fig)
        plt.close(fig)

        if st.button("Predict Emotion", type="primary"):
            # Apply the exact training-set normalization
            model_input = (mfcc - mean.reshape(1, N_MFCC)) / (
                std.reshape(1, N_MFCC)
            )
            model_input = np.expand_dims(model_input, axis=0)

            probabilities = model.predict(
                model_input,
                verbose=0
            )[0]

            predicted_index = int(np.argmax(probabilities))
            predicted_emotion = labels[predicted_index]

            st.subheader("Prediction")
            st.write(f"**Predicted emotion: {predicted_emotion.title()}**")
            st.write(
                f"Model score: {probabilities[predicted_index] * 100:.2f}%"
            )

            st.caption(
                "Scores are model confidence estimates, not calibrated "
                "probabilities that a person truly feels this emotion."
            )

            st.subheader("Scores for all emotions")

            for label, score in sorted(
                zip(labels, probabilities),
                key=lambda item: item[1],
                reverse=True
            ):
                st.write(f"{label.title()}: {score * 100:.2f}%")
                st.progress(float(score))

    except Exception as error:
        st.error(f"Could not analyze this audio: {error}")

    finally:
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)


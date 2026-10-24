# Speech Emotion Recognition Using CNN and LSTM

## Abstract
Speech Emotion Recognition identifies emotion categories from acoustic properties of speech. This project proposes extracting MFCC, pitch and energy features and processing them using CNN and LSTM deep-learning layers.

## Objectives
- Accept recorded speech as input.
- Preprocess audio consistently.
- Extract MFCC, pitch and energy features.
- Design a five-class emotion classifier.
- Provide an interface for audio analysis and future real-time recognition.

## Methodology
1. Upload a speech recording.
2. Convert it to mono and resample it.
3. Extract MFCC, pitch and energy features.
4. Use CNN layers to learn local acoustic patterns.
5. Use LSTM layers to learn temporal dependencies.
6. Use a trained output layer to classify emotions.

## Prototype
The application accepts audio, displays playback, calculates MFCCs, estimates pitch and energy, and visualizes the features. It demonstrates the proposed classification workflow.

## Limitation
The current prototype does not include a trained CNN-LSTM model. Its demo emotion labels and scores are illustrative and are not genuine predictions.

## Evaluation Plan
Train and test on a labelled dataset. Split data by speaker where possible and report accuracy, macro-F1, precision, recall and a confusion matrix.

## Future Scope
Train a real model using a labelled dataset such as RAVDESS or CREMA-D, integrate model weights, and add microphone-based recognition.

## Conclusion
This prototype demonstrates the audio-processing and interface stages. A trained and evaluated CNN-LSTM model is required for genuine emotion classification.

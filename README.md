# 🚨 Real-Time Voice Signal Classification for Emergency Detection

This project implements a deep learning-based system that classifies real-time audio inputs as **emergency (e.g., screams)** or **non-emergency (e.g., laughter, background noise)** using **Convolutional Neural Networks (CNNs)** and **Mel spectrogram analysis**. It includes a Flask-based REST API for real-time detection and automated emergency alerts.

## 🔍 Project Overview

- **Goal**: Detect distress/emergency sounds in real time and trigger alerts.
- **Model**: Custom CNN trained on Mel spectrograms extracted from audio files.
- **Performance**: Achieved **95.68% accuracy**, **96.51% precision**, **95.13% recall**.
- **Deployment**: Real-time classification via a Flask API with alert generation using Twilio.

## 🎯 Key Features

- 🎤 Real-time audio input capture
- 🧠 CNN-based binary audio classification
- 📊 Spectrogram feature extraction using Librosa
- 💬 Flask API for model inference
- 📱 SMS alert system using Twilio
- 🧪 Robust performance with dropout and early stopping
- 📈 Evaluation metrics: Accuracy, Precision, Recall, F1 Score, Confusion Matrix

## 🛠️ Tech Stack

- **Python 3**
- **TensorFlow / Keras**
- **Librosa** – Audio processing
- **NumPy** – Numerical operations
- **Matplotlib** – Visualization
- **Flask** – REST API
- **Twilio** – SMS alerts
- **Scikit-learn** – Evaluation metrics

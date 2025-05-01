from flask import Flask, request, jsonify
from flask_cors import CORS
import librosa
import numpy as np
import tensorflow as tf
import os
from twilio.rest import Client
from pydub import AudioSegment

app = Flask(__name__)
CORS(app)

MODEL_PATH = r"D:/SEM 8/codes/model1.keras"
model = tf.keras.models.load_model(MODEL_PATH)

emergency_contact = "+1234567890"  # Replace with actual number

def process_audio(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    y = librosa.util.normalize(y)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    if mel_spec.shape[1] < 128:
        mel_spec = np.pad(mel_spec, ((0, 0), (0, 128 - mel_spec.shape[1])), mode='constant')
    else:
        mel_spec = mel_spec[:, :128]
    mel_spec = (mel_spec - np.min(mel_spec)) / (np.max(mel_spec) - np.min(mel_spec) + 1e-6)
    mel_spec = np.expand_dims(mel_spec, axis=-1)
    mel_spec = np.repeat(mel_spec, 3, axis=-1)
    return np.expand_dims(mel_spec, axis=0)

@app.route("/")
def home():
    return "Flask server is running!"

@app.route("/predict", methods=["POST"])
def predict():
    global emergency_contact
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    file = request.files['audio']
    filename = "temp.wav"
    file.save(filename)
    if file.filename.endswith(".mp3"):
        sound = AudioSegment.from_mp3(filename)
        sound.export("temp.wav", format="wav")
        filename = "temp.wav"
    spectrogram = process_audio(filename)
    if spectrogram is None:
        return jsonify({"error": "Invalid or silent audio file"}), 400
    prediction = model.predict(spectrogram)[0, 0]
    threshold = 0.7
    result = "emergency" if prediction > threshold else "non-emergency"
    if result == "emergency":
        send_sms(emergency_contact, "Emergency detected! Please check immediately.")
    return jsonify({"prediction": result, "probability": float(prediction)})

@app.route("/set_contact", methods=["POST"])
def set_contact():
    global emergency_contact
    data = request.json
    if "contact" in data:
        emergency_contact = data["contact"]
        return jsonify({"message": "Emergency contact updated!", "contact": emergency_contact})
    return jsonify({"error": "No contact provided"}), 400

def send_sms(contact, message):
    account_sid = "your_twilio_sid"
    auth_token = "your_twilio_auth_token"
    client = Client(account_sid, auth_token)
    try:
        client.messages.create(body=message, from_="+your_twilio_number", to=contact)
    except Exception as e:
        print(f"SMS Error: {e}")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)

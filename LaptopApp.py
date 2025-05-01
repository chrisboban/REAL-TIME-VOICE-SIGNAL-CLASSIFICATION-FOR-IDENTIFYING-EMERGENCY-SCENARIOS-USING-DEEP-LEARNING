import tkinter as tk
from tkinter import messagebox
import sounddevice as sd
import numpy as np
import soundfile as sf
import threading
import requests
import time
import os

FLASK_API_URL = "http://127.0.0.1:5000/predict"

DURATION = 3  # Seconds
SAMPLERATE = 22050
SAVE_PATH = "recordings/"
os.makedirs(SAVE_PATH, exist_ok=True)

listening = False

def record_audio(duration=DURATION, samplerate=SAMPLERATE):
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype=np.float32)
    sd.wait()
    filename = f"{SAVE_PATH}recording_{int(time.time())}.wav"
    sf.write(filename, audio, samplerate)
    return filename

def send_audio(filename):
    try:
        with open(filename, 'rb') as f:
            files = {'audio': f}
            response = requests.post(FLASK_API_URL, files=files)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": "Server returned an error."}
    except Exception as e:
        return {"error": str(e)}

def listen_loop(status_label):
    global listening
    while listening:
        filename = record_audio()
        result = send_audio(filename)
        if "prediction" in result:
            pred = result["prediction"]
            prob = float(result["probability"])
            status_label.config(text=f"{pred.upper()} ({prob:.2f})", fg="red" if pred == "emergency" else "green")
            if pred == "emergency":
                print("Emergency detected! Taking action...")
        else:
            print("Error:", result.get("error"))
        time.sleep(1)

def start_listening(status_label, start_btn, stop_btn):
    global listening
    if not listening:
        listening = True
        start_btn.config(state="disabled")
        stop_btn.config(state="normal")
        threading.Thread(target=listen_loop, args=(status_label,), daemon=True).start()

def stop_listening(status_label, start_btn, stop_btn):
   
::contentReference[oaicite:13]{index=13}
 

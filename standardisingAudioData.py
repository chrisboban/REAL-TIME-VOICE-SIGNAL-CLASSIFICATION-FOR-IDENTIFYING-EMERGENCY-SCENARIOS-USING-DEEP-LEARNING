import librosa
import soundfile as sf
import numpy as np
import os

def format_audio(audio_path, output_path, target_sr=22050, target_duration=4.0):
    """Loads, resamples, normalizes, trims, and pads audio to a fixed duration."""
    
    # Load audio file
    y, sr = librosa.load(audio_path, sr=target_sr, mono=True)
    
    # Trim leading and trailing silence
    # y, _ = librosa.effects.trim(y)

    # Normalize amplitude
    y = librosa.util.normalize(y)
    
    # Set fixed duration (pad or trim)
    target_length = int(target_sr * target_duration)  # Convert seconds to samples
    if len(y) > target_length:
        y = y[:target_length]  # Trim
    else:
        y = np.pad(y, (0, target_length - len(y)))  # Pad with zeros
    
    # Save formatted audio
    sf.write(output_path, y, target_sr)

    return output_path

def preprocess_dataset(input_folder, output_folder):
    """Preprocess all audio files in a folder."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in os.listdir(input_folder):
        if file_name.endswith(".wav"):
            input_path = os.path.join(input_folder, file_name)
            output_path = os.path.join(output_folder, file_name)
            format_audio(input_path, output_path)
            print(f"Processed: {file_name}")

# Example usage
preprocess_dataset("D:/SEM 8/Converted_Separately/scream", "D:/SEM 8/standardised_Audio/scream")
preprocess_dataset("D:/SEM 8/Converted_Separately/non_scream", "D:/SEM 8/standardised_Audio/non_scream")
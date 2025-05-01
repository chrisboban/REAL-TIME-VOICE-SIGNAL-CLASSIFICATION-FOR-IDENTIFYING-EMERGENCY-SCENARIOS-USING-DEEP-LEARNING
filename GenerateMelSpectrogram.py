import librosa 
import librosa.display 
import matplotlib.pyplot as plt 
import numpy as np 
 
import os 
from PIL import Image 
 
def create_spectrogram(audio_file, image_file): 
    fig = plt.figure() 
    ax = fig.add_subplot(1, 1, 1) 
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1) 
 
    y, sr = librosa.load(audio_file, sr=22050) 
     
    #Generate Mel spectrogram 
    ms = librosa.feature.melspectrogram(y=y, sr=sr) 
    log_ms = librosa.power_to_db(ms, ref=np.max) 
     
    librosa.display.specshow(log_ms, sr=sr) 
     
    #Save to temporary location 
    temp_path = 'temp.png' 
    fig.savefig(temp_path) 
    plt.close(fig) 
 
    #Resize the image to 128x128 for uniformity 
    img = Image.open(temp_path) 
    img = img.resize((128, 128)) 
 
    img.save(image_file) 
    os.remove(temp_path) 
 
def create_pngs_from_wavs(input_path, output_path): 
    if not os.path.exists(output_path): 
        os.makedirs(output_path) 
 
    dir = os.listdir(input_path) 
 
 
    for i, file in enumerate(dir): 
        if file.endswith('.wav'): 
            input_file = os.path.join(input_path, file) 
            output_file = os.path.join(output_path, file.replace('.wav', '.png')) 
             
            print(f"Processing file {i+1}/{len(dir)}: {file}") 
            create_spectrogram(input_file, output_file) 
 
input_path = "D:/SEM 8/standardised_Audio/non_scream" 
output_path = "D:/SEM 8/spectrograms/non_scream" 
 
create_pngs_from_wavs(input_path, output_path) 
 
SpectrogramLabelling.py 
import os 
import pandas as pd 
 
def create_csv_for_labels(screams_dir, non_screams_dir, csv_file): 
    spectrogram_paths = [] 
    labels = [] 
 
    for filename in os.listdir(screams_dir): 
        if filename.endswith('.png'): 
            spectrogram_paths.append(os.path.join(screams_dir, filename)) 
            labels.append(1) 
    for filename in os.listdir(non_screams_dir): 
        if filename.endswith('.png'): 
            spectrogram_paths.append(os.path.join(non_screams_dir, filename)) 
            labels.append(0) 
 
    data = {'file_path': spectrogram_paths, 'label': labels} 
    df = pd.DataFrame(data) 
df.to_csv(csv_file, index=False) 
screams_dir = "D:/SEM 8/spectrograms/scream" 
non_screams_dir = "D:/SEM 8/spectrograms/non_scream" 
csv_file = 'spectrogram_labels.csv' 
create_csv_for_labels(screams_dir, non_screams_dir, csv_file)
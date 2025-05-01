import os 
import pandas as pd 
 
def label_audio_files(input_folder, label, output_csv): 
    data = [] 
    for filename in os.listdir(input_folder): 
        if filename.endswith(".wav"): 
            data.append([filename, label]) 
 
    df = pd.DataFrame(data, columns=["filename", "label"]) 
    df.to_csv(output_csv, mode='a', header=not os.path.exists(output_csv), index=False) 
    print(f"Processed files from {input_folder} and added to {output_csv}") 
 
label_audio_files('D:/SEM 8/standardised_Audio/scream', 1, 'labels.csv') 
label_audio_files('D:/SEM 8/standardised_Audio/non_scream', 0, 'labels.csv')
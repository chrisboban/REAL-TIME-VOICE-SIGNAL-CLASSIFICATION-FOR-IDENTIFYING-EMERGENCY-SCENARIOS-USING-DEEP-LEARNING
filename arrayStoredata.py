import pandas as pd
import numpy as np
import os
from tensorflow import keras
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.preprocessing import image
from keras.preprocessing import image




# Step 1: Load the CSV File (NO SHUFFLING)
csv_file = "shuffled_dataset.csv"  #  Replace with your actual CSV path
data = pd.read_csv(csv_file)

# Step 2: Split into Training (80%) and Validation (20%) - No Shuffle
train_size = int(0.8 * len(data))
train_data = data[:train_size]  # First 80% for training
val_data = data[train_size:]   # Last 20% for validation

print(f"Dataset split: {len(train_data)} training samples, {len(val_data)} validation samples")

# Function to Load and Process Spectrograms
def load_data_from_dataframe(data, img_size=(128, 128)):
    images = []
    labels = []

    for index, row in data.iterrows():
        img_path = row['file_path']  # 🔹 Ensure CSV has 'filename' column
        label = row['label']        # 🔹 Ensure CSV has 'label' column (1 = emergency, 0 = non-emergency)

        if os.path.exists(img_path):  # Ensure the file exists before loading
            # Load the image
            img = image.load_img(img_path, target_size=img_size, color_mode='rgb')
            img_array = image.img_to_array(img)

            images.append(img_array)
            labels.append(label)

    X = np.array(images, dtype=np.float32) / 255.0  # Normalize images
    y = np.array(labels, dtype=np.int32)

    return X, y

# Step 3: Load Training & Validation Data
X_train, y_train = load_data_from_dataframe(train_data)
X_val, y_val = load_data_from_dataframe(val_data)

# Step 4: Save Processed Data for Future Use
np.savez("train_data.npz", X=X_train, y=y_train)
np.savez("val_data.npz", X=X_val, y=y_val)

print("Processed training and validation data saved successfully.")

# Step 5: Load the Saved Data (for verification)
data_train = np.load("train_data.npz")
X_train, y_train = data_train["X"], data_train["y"]

data_val = np.load("val_data.npz")
X_val, y_val = data_val["X"], data_val["y"]

print(f"Loaded Data Shapes -> X_train: {X_train.shape}, y_train: {y_train.shape}, X_val: {X_val.shape}, y_val: {y_val.shape}")
# data_preprocessing.py

import os
import json
import numpy as np
from PIL import Image

def load_data(data_dirs):
    images = []
    labels = []
    for data_dir in data_dirs:
        for filename in os.listdir(data_dir):
            if filename.endswith('_face.jpg'):
                image_path = os.path.join(data_dir, filename)
                images.append(np.array(Image.open(image_path)))
                # Extract label from corresponding JSON file
                json_filename = filename.replace('_face.jpg', '_metadata.json')
                json_path = os.path.join(data_dir, json_filename)
                with open(json_path, 'r') as f:
                    metadata = json.load(f)
                label = metadata['label']
                labels.append(1 if label == 'fake' else 0)  # 1 for fake, 0 for real
    return np.array(images), np.array(labels)

def preprocess_images(images):
    # Normalize pixel values to range [-1, 1]
    return (images.astype(np.float32) - 127.5) / 127.5

def preprocess_data(train_dirs, val_dir):
    # Load training data
    x_train, y_train = load_data(train_dirs)
    x_train = preprocess_images(x_train)
    
    # Load validation data
    x_val, y_val = load_data(val_dir)
    x_val = preprocess_images(x_val)
    
    return (x_train, y_train), (x_val, y_val)


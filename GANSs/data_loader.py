import os
import json
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img

class DataLoader:
    def __init__(self, data_dir, img_size, batch_size):
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size

    def load_data(self):
        while True:
            X, y = [], []

            # List all files in the data directory
            files = [file for file in os.listdir(self.data_dir) if file.endswith('_metadata.json')]

            # Shuffle the list of files
            np.random.shuffle(files)

            for file in files:
                with open(os.path.join(self.data_dir, file), 'r') as f:
                    metadata = json.load(f)
                    label = metadata['label']  # 'REAL' or 'FAKE'


                    frame_filename = file.replace('_metadata.json', '_face.jpg')

                    img = load_img(os.path.join(self.data_dir, frame_filename), target_size=self.img_size)
                    img_array = img_to_array(img) / 255.0  # Normalize pixel values
                    X.append(img_array)
                    binary_label = 1 if label == 'REAL' else 0
                    y.append(binary_label)  # Convert labels to binary (REAL=1, FAKE=0)


                    # Yield batch if it reaches batch size
                    if len(X) == self.batch_size:
                        yield np.array(X), np.array(y)
                        X, y = [], []

            # Yield the remaining data as the last batch
            if X:
                yield np.array(X), np.array(y)

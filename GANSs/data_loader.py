import os
import json
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img

class DataLoader:
    def __init__(self, data_dir, img_size, batch_size):
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size

        self.files = [file for file in os.listdir(self.data_dir) if file.endswith('_metadata.json')]
        self.num_files = len(self.files)
        self.samples_per_epoch = self.num_files * 10 
        self.total_samples = self.num_files * 10  
        self.current_index = 0

    def load_data(self):
        samples_processed = 0  # Counter to track the number of samples processed

        while True:
            if self.current_index >= self.total_samples:
                self.current_index = 0  # Reset index if end of dataset is reached
                np.random.shuffle(self.files)  # Shuffle dataset between epochs

            X, y = [], []

            for _ in range(self.batch_size):
                if self.current_index >= self.total_samples:
                    break  # Stop iteration if end of dataset is reached

                file = self.files[self.current_index // 10]  # Index to metadata file

                with open(os.path.join(self.data_dir, file), 'r') as f:
                    metadata = json.load(f)
                    label = metadata['label']  # 'REAL' or 'FAKE'

                    frame_filename = file.replace('_metadata.json', '_face.jpg')

                    img = load_img(os.path.join(self.data_dir, frame_filename), target_size=self.img_size)
                    img_array = img_to_array(img) / 255.0  # Normalize pixel values
                    X.append(img_array)
                    binary_label = 1 if label == 'REAL' else 0
                    y.append(binary_label)  # Convert labels to binary (REAL=1, FAKE=0)

                self.current_index += 1
                samples_processed += 1  # Increment samples_processed counter

                # Print feedback during training
                print(f"Processed {samples_processed} out of {self.total_samples} samples.")

            if not X:
                break  # Stop iteration if no more samples are available

            yield np.array(X), np.array(y)

# evaluate.py

import numpy as np
from tensorflow.keras.models import load_model
from data_preprocessing import preprocess_data

# Parameters
input_shape = (64, 64, 3)
model_path = 'path/to/saved/discriminator/model.h5'
val_dir = 'path/to/validation/dir'

# Load validation data
(x_val, y_val) = preprocess_data(val_dir)

# Load the trained discriminator model
discriminator = load_model(model_path)

# Evaluate the model
loss, accuracy = discriminator.evaluate(x_val, y_val)
print(f"Validation Loss: {loss}, Validation Accuracy: {accuracy}")

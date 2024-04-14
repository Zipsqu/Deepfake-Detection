import numpy as np
from data_loader import DataLoader
from discriminator import build_discriminator
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Parameters
train_data_dir = 'D:/Dataset/Training'
val_data_dir = 'D:/Dataset/Validation'
img_size = (64, 64)
batch_size = 32
epochs = 10
patience = 3  # Number of epochs with no improvement after which training will be stopped
discriminator_save_path = 'D:/Dataset/discriminator_model.h5'

print("Creating data loaders...")
# Create data loaders
train_data_loader = DataLoader(train_data_dir, img_size, batch_size)
val_data_loader = DataLoader(val_data_dir, img_size, batch_size)

print("Building and compiling discriminator...")
# Build and compile discriminator
discriminator = build_discriminator(input_shape=img_size + (3,))
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5), metrics=['accuracy'])

# Enable GPU usage with CUDA
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_accuracy', patience=patience, verbose=1, restore_best_weights=True)

print("Training discriminator...")
# Train discriminator with early stopping
discriminator.fit(train_data_loader.load_data(),
                   validation_data=val_data_loader.load_data(),
                   batch_size=batch_size,
                   epochs=epochs,
                   callbacks=[early_stopping],
                   verbose=1)

# Save trained discriminator model
print("Saving trained discriminator model...")
discriminator.save(discriminator_save_path)

print("Training complete.")

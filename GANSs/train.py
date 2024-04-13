# train.py

import numpy as np
import tensorflow as tf
from model import build_generator, build_discriminator
from data_preprocessing import preprocess_data

# Parameters
input_shape = (64, 64, 3)
latent_dim = 100
epochs = 100
batch_size = 64
model_save_path = 'D:/Dataset/model.h5'  # Specify the path to save the model
train_dirs = ['D:/Dataset/Training']
val_dir = 'D:/Dataset/Validation'

# Build and compile the discriminator
discriminator = build_discriminator(input_shape)
discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Build the generator
generator = build_generator((latent_dim,))
fake_image = generator(tf.keras.Input(shape=(latent_dim,)))

# The combined model (stacked generator and discriminator)
discriminator.trainable = False
validity = discriminator(fake_image)
combined = tf.keras.Model(generator.input, validity)
combined.compile(loss='binary_crossentropy', optimizer='adam')

# Preprocess data
(x_train, y_train), (x_val, y_val) = preprocess_data(train_dirs, val_dir)

# Training loop
for epoch in range(epochs):
    # Shuffle training data
    indices = np.random.permutation(len(x_train))
    x_train_shuffled = x_train[indices]
    y_train_shuffled = y_train[indices]
    
    # Mini-batch training
    for i in range(0, len(x_train), batch_size):
        batch_images = x_train_shuffled[i:i+batch_size]
        batch_labels = y_train_shuffled[i:i+batch_size]
        
        # Train discriminator
        d_loss = discriminator.train_on_batch(batch_images, batch_labels)
        
        # Train generator (combined model)
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        valid_labels = np.ones(batch_size)
        g_loss = combined.train_on_batch(noise, valid_labels)
    
    # Evaluate on validation data
    val_loss, val_accuracy = discriminator.evaluate(x_val, y_val, verbose=0)
    print(f"Epoch {epoch+1}/{epochs}, Discriminator Validation Loss: {val_loss}, Discriminator Validation Accuracy: {val_accuracy}")

# Save the trained discriminator model
discriminator.save(model_save_path)

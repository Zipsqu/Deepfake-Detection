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
model_save_path = 'path/to/save/discriminator/model.h5'  # Specify the path to save the model

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
train_dirs = ['path/to/real/training/dir', 'path/to/fake/training/dir']
val_dir = 'path/to/validation/dir'

(x_train, y_train), (x_val, y_val) = preprocess_data(train_dirs, val_dir)

# Training loop
for epoch in range(epochs):
    # Shuffle training data
    indices = np.random.permutation(len(x_train))
    x_train_shuffled = x_train[indices]
    y_train_shuffled = y_train[indices]
    
    # Mini-batch training
    for i in range(0, len(x_train), batch_size):
        real_images = x_train_shuffled[i:i+batch_size]
        real_labels = y_train_shuffled[i:i+batch_size]
        
        # Train discriminator on real images
        d_loss_real = discriminator.train_on_batch(real_images, real_labels)
        
        # Train discriminator on fake images
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        fake_images = generator.predict(noise)
        fake_labels = np.zeros(batch_size)
        d_loss_fake = discriminator.train_on_batch(fake_images, fake_labels)
        
        # Train generator (combined model)
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        valid_labels = np.ones(batch_size)
        g_loss = combined.train_on_batch(noise, valid_labels)

# Save the trained discriminator model
discriminator.save(model_save_path)

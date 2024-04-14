import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense

def build_discriminator(input_shape):
    model = Sequential([
        Conv2D(64, (3, 3), strides=(2, 2), padding='same', input_shape=input_shape, activation='relu'),
        Conv2D(128, (3, 3), strides=(2, 2), padding='same', activation='relu'),
        Flatten(),
        Dense(1, activation='sigmoid')
    ])
    return model

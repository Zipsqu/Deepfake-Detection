import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape

def build_generator(input_shape=(100,)):
    model = Sequential([
        Dense(256, input_shape=input_shape, activation='relu'),
        Dense(512, activation='relu'),
        Dense(1024, activation='relu'),
        Dense(64 * 64 * 3, activation='sigmoid'),  # Output layer
        Reshape((64, 64, 3))
    ])
    return model

import os
import json
import tensorflow as tf
import albumentations as A
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set GPU memory growth to avoid memory allocation issues
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Define paths to dataset
train_dir = 'path/to/training/data'
val_dir = 'path/to/validation/data'

# Define image size and batch size
img_size = (224, 224)
batch_size = 32

# Define data augmentation transforms using Albumentations
train_augmentation = A.Compose([
    A.Rotate(limit=20, p=0.5),
    A.RandomResizedCrop(img_size[0], img_size[1], scale=(0.8, 1.0), p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.HueSaturationValue(p=0.2)
])

# No data augmentation for validation dataset
val_augmentation = A.Compose([])

# Define data loading functions
def load_image(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, img_size)
    img = tf.cast(img, tf.float32) / 255.0
    return img

def process_data(image, label):
    return image, label

# Load training and validation datasets using tf.data API
train_dataset = tf.data.Dataset.list_files(os.path.join(train_dir, '*/*.jpg'))
train_dataset = train_dataset.map(lambda x: (load_image(x), int(os.path.dirname(x).split('/')[-1])))
train_dataset = train_dataset.map(lambda x, y: (train_augmentation(image=x)['image'], y))
train_dataset = train_dataset.map(process_data)
train_dataset = train_dataset.shuffle(buffer_size=1000).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

val_dataset = tf.data.Dataset.list_files(os.path.join(val_dir, '*/*.jpg'))
val_dataset = val_dataset.map(lambda x: (load_image(x), int(os.path.dirname(x).split('/')[-1])))
val_dataset = val_dataset.map(lambda x, y: (val_augmentation(image=x)['image'], y))
val_dataset = val_dataset.map(process_data)
val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

# Load EfficientNetB0 model with pre-trained weights
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add custom head to the model
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(256, activation='relu')(x)
x = Dense(1, activation='sigmoid')(x)

# Compile the model
model = Model(inputs=base_model.input, outputs=x)
model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_dataset,
    epochs=10,
    validation_data=val_dataset,
    callbacks=[ModelCheckpoint('efficientnetb0_model.h5', save_best_only=True, monitor='val_loss')]
)

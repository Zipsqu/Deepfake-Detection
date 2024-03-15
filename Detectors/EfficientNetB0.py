import os
import json
import tensorflow as tf
import albumentations as A
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
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
train_dir = "D:/DFDC Sample Dataset/Pre-processed Dataset/Training"
val_dir = 'D:/DFDC Sample Dataset/Pre-processed Dataset/Validation'

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
def load_data(file_path):
    # Load image
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, img_size)
    img = tf.cast(img, tf.float32) / 255.0

    # Load JSON file and extract label and landmarks
    json_path = tf.strings.regex_replace(file_path, '_face.jpg', '_metadata.json')
    json_data = tf.io.read_file(json_path)
    data = tf.io.decode_json_example(json_data)
    label = tf.cond(tf.equal(data['label'], 'FAKE'), lambda: tf.constant(1), lambda: tf.constant(0))  # Assign label based on JSON content

    return img, label


# Load training and validation datasets using tf.data API
train_dataset = tf.data.Dataset.list_files(os.path.join(train_dir, '*_face.jpg'))
train_dataset = train_dataset.map(load_data)
train_dataset = train_dataset.map(lambda x, y: (train_augmentation(image=x)['image'], y))
train_dataset = train_dataset.shuffle(buffer_size=1000).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

val_dataset = tf.data.Dataset.list_files(os.path.join(val_dir, '*_face.jpg'))
val_dataset = val_dataset.map(load_data)
val_dataset = val_dataset.map(lambda x, y: (val_augmentation(image=x)['image'], y))
val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

# Load EfficientNetB0 model with pre-trained weights
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add custom head to the model
input_tensor = Input(shape=(224, 224, 3))
x = train_augmentation(image=input_tensor)['image']  # Apply data augmentation
x = base_model(x)
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
output_1 = Dense(1, activation='sigmoid', name='classification_output')(x)
output_2 = Dense(10, name='landmarks_output')(x)  # Assuming landmarks is a 10-dimensional vector
model = Model(inputs=input_tensor, outputs=[output_1, output_2])

# Compile the model
model.compile(optimizer=Adam(lr=0.0001),
              loss={'classification_output': 'binary_crossentropy', 'landmarks_output': 'mse'},
              metrics={'classification_output': 'accuracy'})

# Train the model
history = model.fit(
    train_dataset,
    epochs=10,
    validation_data=val_dataset,
    callbacks=[ModelCheckpoint('D:/DFDC Sample Dataset/efficientnetb0_model.h5', save_best_only=True,
                               monitor='val_classification_output_loss')]
)

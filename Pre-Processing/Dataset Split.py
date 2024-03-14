import os
import shutil
import random

# Define paths to the directory containing images and metadata JSON files
images_dir = 'path/to/images'
metadata_dir = 'path/to/metadata'

# Define paths to the directories where the separated data sets will be stored
train_dir = 'path/to/train'
val_dir = 'path/to/val'
test_dir = 'path/to/test'

# Define the split ratios for train, validation, and test sets (e.g., 80%, 10%, 10%)
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

# Create directories for train, validation, and test sets if they do not exist
for directory in [train_dir, val_dir, test_dir]:
    os.makedirs(directory, exist_ok=True)

# Get the list of all image files
image_files = os.listdir(images_dir)

# Shuffle the list of image files
random.shuffle(image_files)

# Group images by video name
video_images = {}
for image in image_files:
    video_name = image.split('_frame_')[0]
    if video_name not in video_images:
        video_images[video_name] = []
    video_images[video_name].append(image)

# Split the videos into train, validation, and test sets
video_list = list(video_images.keys())
num_videos = len(video_list)
num_train = int(train_ratio * num_videos)
num_val = int(val_ratio * num_videos)
num_test = num_videos - num_train - num_val

# Split videos into train, validation, and test sets
train_videos = video_list[:num_train]
val_videos = video_list[num_train:num_train+num_val]
test_videos = video_list[num_train+num_val:]

# Move images to the respective directories along with their associated metadata
def move_files(videos, source_dir, dest_dir):
    for video in videos:
        for image in video_images[video]:
            image_name, _ = os.path.splitext(image)
            metadata_file = image_name + '_metadata.json'
            shutil.move(os.path.join(source_dir, image), os.path.join(dest_dir, image))
            shutil.move(os.path.join(metadata_dir, metadata_file), os.path.join(dest_dir, metadata_file))

move_files(train_videos, images_dir, train_dir)
move_files(val_videos, images_dir, val_dir)
move_files(test_videos, images_dir, test_dir)

print("Data separation completed.")

import os
import shutil
import random

# Define paths to the directory containing images and metadata JSON files
source_dir = 'path/to/source_directory'  # Change this to the actual path

# Define paths to the directories where the separated data sets will be stored
train_dir = 'path/to/train'  # Change this to the actual path
val_dir = 'path/to/val'  # Change this to the actual path
test_dir = 'path/to/test'  # Change this to the actual path

# Define the split ratios for train, validation, and test sets (e.g., 80%, 10%, 10%)
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

# Create directories for train, validation, and test sets if they do not exist
for directory in [train_dir, val_dir, test_dir]:
    os.makedirs(directory, exist_ok=True)

# Get the list of all image files
image_files = [filename for filename in os.listdir(source_dir) if filename.endswith('.jpg')]

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

# Move images and their associated metadata files to the respective directories
def move_files(videos, source_dir, dest_dir):
    for video in videos:
        for image in video_images[video]:
            image_name, _ = os.path.splitext(image)
            image_path = os.path.join(source_dir, image)
            metadata_file = image_name + '_metadata.json'
            metadata_path = os.path.join(source_dir, metadata_file)
            dst_image_path = os.path.join(dest_dir, image)
            dst_metadata_path = os.path.join(dest_dir, metadata_file)
            try:
                shutil.move(image_path, dst_image_path)
                shutil.move(metadata_path, dst_metadata_path)
                print(f"Moved files: {image_path} to {dst_image_path}, {metadata_path} to {dst_metadata_path}")
            except FileNotFoundError as e:
                print(f"Error: {e}")
                print(f"File not found: {image_path} or {metadata_path}")

# Move files for each dataset
move_files(train_videos, source_dir, train_dir)
move_files(val_videos, source_dir, val_dir)
move_files(test_videos, source_dir, test_dir)

print("Data separation completed.")

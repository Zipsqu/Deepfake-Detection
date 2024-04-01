import os
import shutil
import random

# Define paths to the directory containing images and metadata JSON files.
source_dir = "D:/DFDC Sample Dataset/Resized Frames"  

# Define path to store splitted dataset.
train_dir = "D:/DFDC Sample Dataset/Pre-processed Dataset/Training"  
val_dir = "D:/DFDC Sample Dataset/Pre-processed Dataset/Validation"  
test_dir = "D:/DFDC Sample Dataset/Pre-processed Dataset/Testing"  

# Define the split ratios.
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

# Create directories if they do not exist.
for directory in [train_dir, val_dir, test_dir]:
    os.makedirs(directory, exist_ok=True)

# Make a list of images and shuffle.
image_files = [filename for filename in os.listdir(source_dir) if filename.endswith('.jpg')]
random.shuffle(image_files)

# Group frames by the video.
video_images = {}
for image in image_files:
    video_name = image.split('_frame_')[0]
    if video_name not in video_images:
        video_images[video_name] = []
    video_images[video_name].append(image)

# Doing the math.
video_list = list(video_images.keys())
num_videos = len(video_list)
num_train = int(train_ratio * num_videos)
num_val = int(val_ratio * num_videos)
num_test = num_videos - num_train - num_val

# Assigning the split.
train_videos = video_list[:num_train]
val_videos = video_list[num_train:num_train+num_val]
test_videos = video_list[num_train+num_val:]

# Moving images and metadata.
def move_files(videos, source_dir, dest_dir):
    for video in videos:
        for image in video_images[video]:
            image_name, _ = os.path.splitext(image)
            image_path = os.path.join(source_dir, image)
            metadata_file = image_name.replace('_face.jpg', '_metadata.json')
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

move_files(train_videos, source_dir, train_dir)
move_files(val_videos, source_dir, val_dir)
move_files(test_videos, source_dir, test_dir)

print("Data separation completed.")

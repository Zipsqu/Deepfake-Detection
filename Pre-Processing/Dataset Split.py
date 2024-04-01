import os
import random
import shutil
from collections import defaultdict

# Set the directory containing images and JSON files
data_dir = 'D:/Dataset/Processed Frames'

# Set the directories for training, validation, and evaluation sets
train_dir = 'D:/Dataset/Training'
val_dir = 'D:/Dataset/Validation'
eval_dir = 'D:/Dataset/Testing'

# Ensure the directories exist, create them if necessary
for directory in [train_dir, val_dir, eval_dir]:
    os.makedirs(directory, exist_ok=True)

# List all image files in the data directory
image_files = [file for file in os.listdir(data_dir) if file.endswith('.jpg')]

# Group images by video names
video_frames = defaultdict(list)
for image_file in image_files:
    video_name, _, _ = image_file.rpartition('_frame_')
    video_frames[video_name].append(image_file)

# Shuffle the list of video names
video_names = list(video_frames.keys())
random.shuffle(video_names)

# Calculate split sizes based on the number of videos
total_videos = len(video_names)
train_size = int(0.8 * total_videos)
val_size = int(0.1 * total_videos)
eval_size = total_videos - train_size - val_size

# Split video names
train_videos = video_names[:train_size]
val_videos = video_names[train_size:train_size + val_size]
eval_videos = video_names[train_size + val_size:]

# Move images and JSON files to respective directories
def move_files(video_list, target_dir):
    for video_name in video_list:
        frames = video_frames[video_name]
        for frame in frames:
            # Get the corresponding JSON file
            json_file = frame.replace('_face.jpg', '_metadata.json')
            # Move image file
            shutil.move(os.path.join(data_dir, frame), os.path.join(target_dir, frame))
            # Move JSON file
            shutil.move(os.path.join(data_dir, json_file), os.path.join(target_dir, json_file))

move_files(train_videos, train_dir)
move_files(val_videos, val_dir)
move_files(eval_videos, eval_dir)

print("Data splitting and moving completed successfully.")

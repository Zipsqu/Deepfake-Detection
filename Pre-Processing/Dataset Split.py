import os
import random
import shutil
from collections import defaultdict

#Set all the directories (with the data_dir being the final directory).
data_dir = 'D:/Dataset/Processed Frames'
train_dir = 'D:/Dataset/Training'
val_dir = 'D:/Dataset/Validation'
eval_dir = 'D:/Dataset/Testing'

# Create directories if they don't exist
for directory in [train_dir, val_dir, eval_dir]:
    os.makedirs(directory, exist_ok=True)

# Group images by video names
video_frames = defaultdict(list)
for file in os.listdir(data_dir):
    if file.endswith('.jpg'):
        video_name, _, _ = file.rpartition('_frame_')
        video_frames[video_name].append(file)

# Shuffle the video names
video_names = list(video_frames.keys())
random.shuffle(video_names)

# Split videos into training, validation, and testing sets
split_sizes = [0.8, 0.1, 0.1]
total_videos = len(video_names)
split_points = [int(size * total_videos) for size in split_sizes]
train_videos, val_videos, eval_videos = video_names[:split_points[0]], video_names[split_points[0]:split_points[0]+split_points[1]], video_names[split_points[0]+split_points[1]:]

# Move files to respective directories
def move_files(video_list, target_dir):
    for video_name in video_list:
        frames = video_frames[video_name]
        for frame in frames:
            json_file = frame.replace('_face.jpg', '_metadata.json')
            shutil.move(os.path.join(data_dir, frame), os.path.join(target_dir, frame))
            shutil.move(os.path.join(data_dir, json_file), os.path.join(target_dir, json_file))

move_files(train_videos, train_dir)
move_files(val_videos, val_dir)
move_files(eval_videos, eval_dir)

print("Data splitting and moving completed.")

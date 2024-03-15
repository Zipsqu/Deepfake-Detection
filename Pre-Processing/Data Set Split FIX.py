import os
import shutil

# Define the source directory containing the JSON files
source_dir = 'D:/DFDC Detector/Extracted Frames 2.0'  # Change this to the actual path

# Define the destination directories where JSON files will be moved
destination_dirs = {
    'train': 'D:/DFDC Detector/Complete Dataset/Training',  # Change this to the actual path
    'val': 'D:/DFDC Detector/Complete Dataset/Validation',      # Change this to the actual path
    'test': 'D:/DFDC Detector/Complete Dataset/Testing'     # Change this to the actual path
}

# Iterate over destination directories
for subset, subset_dir in destination_dirs.items():
    # Iterate through the images in the destination directory
    for filename in os.listdir(subset_dir):
        if filename.endswith('_face.jpg'):
            # Extract the video name from the image filename
            video_name = filename.replace('_face.jpg', '')
            # Construct the corresponding JSON filename
            json_filename = video_name + '_metadata.json'
            # Check if the JSON file exists in the source directory
            json_path = os.path.join(source_dir, json_filename)
            if os.path.exists(json_path):
                # Move the JSON file to the same directory as the image
                dest_path = os.path.join(subset_dir, json_filename)
                shutil.move(json_path, dest_path)
                print(f"Moved JSON file: {json_path} to {dest_path}")
            else:
                print(f"No matching JSON file found for image: {filename}")

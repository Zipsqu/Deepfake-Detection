import os
import shutil
import json

def move_json_files(source_dir, destination_dir):
    # Iterate through files in the source directory
    for filename in os.listdir(source_dir):
        if filename.endswith('.json'):
            json_path = os.path.join(source_dir, filename)
            # Load JSON data
            with open(json_path, 'r') as f:
                metadata = json.load(f)

            # Assuming images were resized to 224x224
            original_width, original_height = 224, 224

            # Adjust bounding box
            bbox = metadata['bounding_box']
            bbox[0] *= (224 / original_width)
            bbox[1] *= (224 / original_height)
            bbox[2] *= (224 / original_width)
            bbox[3] *= (224 / original_height)

            # Adjust landmark coordinates
            landmarks = metadata['landmarks']
            for landmark, coordinates in landmarks.items():
                coordinates[0] *= (224 / original_width)
                coordinates[1] *= (224 / original_height)

            # Move JSON file to the destination directory
            shutil.move(json_path, destination_dir)

# Example usage
source_directory = "D:/DFDC Detector/Extracted Frames"
destination_directory = "D:/DFDC Detector/Extracted Frames 2.0"
move_json_files(source_directory, destination_directory)

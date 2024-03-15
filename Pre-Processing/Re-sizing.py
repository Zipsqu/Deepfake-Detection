import os
import shutil
import json
from PIL import Image

def resize_images(input_dir, output_dir):
    # Iterate through files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith('.jpg'):
            image_path = os.path.join(input_dir, filename)
            basename, ext = os.path.splitext(filename)
            
            # Load image
            image = Image.open(image_path)
            
            # Resize image
            resized_image = image.resize((224, 224))
            
            # Save resized image
            resized_image_path = os.path.join(output_dir, f"{basename}.jpg")
            resized_image.save(resized_image_path)
            
            # Find associated metadata file
            metadata_filename = f"{basename}_metadata.json"
            metadata_path = os.path.join(input_dir, metadata_filename)
            if os.path.exists(metadata_path):
                # Load metadata
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # Adjust bounding box
                bbox = metadata['bounding_box']
                bbox[0] *= (224 / image.width)
                bbox[1] *= (224 / image.height)
                bbox[2] *= (224 / image.width)
                bbox[3] *= (224 / image.height)
                
                # Adjust landmark coordinates
                landmarks = metadata['landmarks']
                for landmark, coordinates in landmarks.items():
                    coordinates[0] *= (224 / image.width)
                    coordinates[1] *= (224 / image.height)
                
                # Save adjusted metadata
                new_metadata_path = os.path.join(output_dir, f"{basename}_metadata.json")
                with open(new_metadata_path, 'w') as f:
                    json.dump(metadata, f)
                # Move metadata file
                shutil.move(metadata_path, new_metadata_path)

# Example usage
input_directory = "path/to/input_directory"
output_directory = "path/to/output_directory"
resize_images(input_directory, output_directory)

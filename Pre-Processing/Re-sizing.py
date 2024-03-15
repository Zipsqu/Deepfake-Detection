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

# Example usage
input_directory = "D:/DFDC Sample Dataset/Extracted Frames"
output_directory = "D:/DFDC Sample Dataset/Resized Frames"
resize_images(input_directory, output_directory)

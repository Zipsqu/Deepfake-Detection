import os
import cv2
import shutil

# Define the directory containing the images
input_dir = 'path/to/input/images'

# Define the directory containing the JSON files
json_dir = 'path/to/json/files'

# Define the output directory to save resized images
output_dir = 'path/to/output/resized_images'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Define the target size for resizing
target_size = (224, 224)

# Iterate over the images in the input directory
for filename in os.listdir(input_dir):
    # Read the image
    image_path = os.path.join(input_dir, filename)
    image = cv2.imread(image_path)

    # Resize the image to the target size
    resized_image = cv2.resize(image, target_size)

    # Save the resized image to the output directory
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, resized_image)

    print(f"Resized {filename} and saved to {output_path}")

    # Move the associated JSON file
    json_filename = filename.replace('_face', '_metadata')
    json_path = os.path.join(json_dir, json_filename)
    if os.path.exists(json_path):
        shutil.copy(json_path, output_dir)
        print(f"Moved {json_filename} to {output_dir}")

print("All images resized successfully.")

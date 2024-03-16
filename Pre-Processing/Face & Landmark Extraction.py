import os
import cv2
from mtcnn import MTCNN
import json
import tensorflow as tf

# Set CUDA_VISIBLE_DEVICES to specify the GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Configure TensorFlow session
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

def extract_frames_and_detect_faces(video_path, output_folder, metadata):
    # Initialize MTCNN for face detection
    detector = MTCNN()

    # Open video file
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    # Extract video name from the file path
    video_name = os.path.basename(video_path)

    # Get metadata for the current video
    video_metadata = metadata.get(video_name, {})
    label = video_metadata.get('label')

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # Detect faces in the frame
        result = detector.detect_faces(frame)
        if result:
            # Extract the bounding box and landmarks
            bounding_box = result[0]['box']
            keypoints = result[0]['keypoints']

            # Crop face using bounding box
            x, y, w, h = bounding_box
            face = frame[y:y + h, x:x + w]

            # Save the cropped face
            output_file = os.path.join(output_folder, f"{video_name}_frame_{frame_count + 1}_face.jpg")
            cv2.imwrite(output_file, face)

            # Associate metadata with the frame
            frame_metadata = {'video_name': video_name, 'frame_index': frame_count + 1, 'label': label,
                              'bounding_box': bounding_box, 'landmarks': keypoints}

            # Save metadata to a JSON file
            metadata_file = os.path.join(output_folder, f"{video_name}_frame_{frame_count + 1}_metadata.json")
            with open(metadata_file, 'w') as f:
                json.dump(frame_metadata, f)

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()


# Metadata file path
metadata_file_path = 'D:/DFDC Sample Dataset/Videos/metadata.json'

# Check if metadata file exists
if os.path.exists(metadata_file_path):
    # Load metadata from JSON file
    with open(metadata_file_path, 'r') as f:
        metadata = json.load(f)

    # Path to directory containing videos
    video_dir = 'D:/DFDC Sample Dataset/Videos'

    # Path to directory where extracted frames will be saved
    output_dir = 'D:/DFDC Sample Dataset/Extracted Frames'

    # Iterate over videos in the directory
    for video_file in os.listdir(video_dir):
        video_path = os.path.join(video_dir, video_file)
        extract_frames_and_detect_faces(video_path, output_dir, metadata)
else:
    print("Metadata file not found at:", metadata_file_path)

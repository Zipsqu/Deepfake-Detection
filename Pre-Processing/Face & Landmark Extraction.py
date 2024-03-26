import os
import cv2
from mtcnn import MTCNN
import json
from multiprocessing import Process, freeze_support

metadata_file_path = 'D:/Dataset/dfdc_train_part_0/metadata.json'
video_dir = 'D:/Dataset/dfdc_train_part_0'
output_dir = 'D:/Dataset/Processed Frames'
detector = MTCNN()

def extract_frames_and_detect_faces(video_path, output_folder, metadata):
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

def process_video(video_path, output_folder, metadata):
    # Wrapper function to call extract_frames_and_detect_faces
    extract_frames_and_detect_faces(video_path, output_folder, metadata)

if __name__ == '__main__':
    # Call freeze_support() if needed
    freeze_support()

    # Check if metadata file exists
    if os.path.exists(metadata_file_path):
        # Load metadata from JSON file
        with open(metadata_file_path, 'r') as f:
            metadata = json.load(f)

        # List to store processes
        processes = []

        # Iterate over videos in the directory
        for video_file in os.listdir(video_dir):
            video_path = os.path.join(video_dir, video_file)
            p = Process(target=process_video, args=(video_path, output_dir, metadata))
            p.start()
            processes.append(p)

        # Wait for all processes to finish
        for p in processes:
            p.join()

    else:
        print("Metadata file not found at:", metadata_file_path)

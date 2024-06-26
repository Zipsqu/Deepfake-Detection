import os
import cv2
from mtcnn import MTCNN
import json
from multiprocessing import Process, cpu_count

# Dataset location, output directory, path to original JSON file.
video_dir = 'D:/Dataset/dfdc_train_part_0'
output_dir = 'D:/Dataset/Processed Frames'
metadata_file_path = 'D:/Dataset/dfdc_train_part_0/metadata.json'

# Double the processes in relation to the available logical processors.
max_processes = cpu_count() * 2
processes = []
detector = MTCNN()

# Opening a video -> Detecting face & Landmarks per frame-> Saving it & related metadata
def extractor_detector(video_path, output_folder, metadata):

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    video_name = os.path.basename(video_path)

    video_metadata = metadata.get(video_name, {})
    label = video_metadata.get('label')

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        result = detector.detect_faces(frame)
        if result:
        
            bounding_box = result[0]['box']
            keypoints = result[0]['keypoints']
            x, y, w, h = bounding_box
            face = frame[y:y + h, x:x + w]

            output_file = os.path.join(output_folder, f"{video_name}_frame_{frame_count + 1}_face.jpg")
            cv2.imwrite(output_file, face)

            frame_metadata = {'video_name': video_name, 'frame_index': frame_count + 1, 'label': label,
                              'bounding_box': bounding_box, 'landmarks': keypoints}

            metadata_file = os.path.join(output_folder, f"{video_name}_frame_{frame_count + 1}_metadata.json")
            with open(metadata_file, 'w') as f:
                json.dump(frame_metadata, f)

        frame_count += 1

    cap.release()


# Wrapper for previous extractor & fixing performance que.
def process_video(video_path, output_folder, metadata):
    
   extractor_detector(video_path, output_folder, metadata)

if __name__ == '__main__':

    if os.path.exists(metadata_file_path):

        with open(metadata_file_path, 'r') as f:
            metadata = json.load(f)

        for video_file in os.listdir(video_dir):
            video_path = os.path.join(video_dir, video_file)
            p = Process(target=process_video, args=(video_path, output_dir, metadata))
            p.start()
            processes.append(p)

            if len(processes) >= max_processes:
                for p in processes:
                    p.join()
                processes = []

        for p in processes:
            p.join()

    else:
        print("Metadata file not found at:", metadata_file_path)

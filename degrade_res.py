import cv2
import os
import numpy as np
from utils import frame_generator  # Import your frame generator module

# Input and Output directories
input_folder = (
    r"C:\Users\BhavyaSehgal\Downloads\bhavya_1st_sem\humaneva\HumanEva\HumanEva"
)
output_folder = (
    r"C:\Users\BhavyaSehgal\Downloads\bhavya_1st_sem\humaneva\HumanEva\degraded_videos"
)
os.makedirs(output_folder, exist_ok=True)


def process_video(input_path, output_path, brightness_factor=50):
    # Open video file to get properties
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error opening video: {input_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Original width
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Original height
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Get original FPS
    cap.release()  # Release since we are using the generator

    # Set target resolution to 720p
    target_width, target_height = 1280, 720

    # Initialize VideoWriter with original FPS
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec
    out = cv2.VideoWriter(output_path, fourcc, fps, (target_width, target_height))

    for frame in frame_generator(input_path):
        # Resize frame to 720p
        frame = cv2.resize(
            frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR
        )

        # Convert to HSV and reduce brightness
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.subtract(v, brightness_factor)  # Reduce Value channel
        hsv = cv2.merge((h, s, v))
        frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        out.write(frame)  # Write the processed frame

    out.release()


def process_all_videos(input_folder, output_folder):
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith((".mp4", ".avi", ".mov", ".mkv")):
                input_path = os.path.join(root, file)
                relative_path = os.path.relpath(input_path, input_folder)
                output_path = os.path.join(output_folder, relative_path)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                print(f"Processing: {input_path} -> {output_path}")
                process_video(input_path, output_path)


# Run processing
process_all_videos(input_folder, output_folder)

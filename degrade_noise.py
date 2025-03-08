import cv2
import os

# Input and Output directories
input_folder = r"C:\Users\BhavyaSehgal\Downloads\bhavya_1st_sem\humaneva\HumanEva\HumanEva"
output_folder = r"C:\Users\BhavyaSehgal\Downloads\bhavya_1st_sem\humaneva\HumanEva\dergarded_videos"
os.makedirs(output_folder, exist_ok=True)

def process_video(input_path, output_path, width=640, height=360, brightness_factor=-40):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error opening video: {input_path}")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Get original FPS
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame
        frame = cv2.resize(frame, (width, height))

        # Decrease brightness
        frame = cv2.add(frame, (brightness_factor, brightness_factor, brightness_factor, 0))

        out.write(frame)

    cap.release()
    out.release()

def process_all_videos(input_folder, output_folder):
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith(('.mp4', '.avi', '.mov', '.mkv')):  # Add more formats if needed
                input_path = os.path.join(root, file)
                relative_path = os.path.relpath(input_path, input_folder)
                output_path = os.path.join(output_folder, relative_path)

                # Create output subfolder if needed
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                print(f"Processing: {input_path} -> {output_path}")
                process_video(input_path, output_path)

# Run processing
process_all_videos(input_folder, output_folder)

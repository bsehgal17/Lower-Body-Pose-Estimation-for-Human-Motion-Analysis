import cv2
import os
import numpy as np
from utils import frame_generator  # Make sure this exists and works correctly

# Input and Output directories
input_folder = (
    r"C:\Users\BhavyaSehgal\Downloads\bhavya_1st_sem\humaneva\HumanEva\HumanEva"
)
output_folder = r"C:\Users\BhavyaSehgal\Downloads\bhavya_1st_sem\humaneva\HumanEva\degraded_videos_20"
os.makedirs(output_folder, exist_ok=True)

# --- Modern Noise Simulation Functions ---


def add_realistic_noise(image):
    # 1. Poisson noise (shot noise)
    noisy = np.random.poisson(image.astype(np.float32)).clip(0, 255).astype(np.uint8)

    # 2. Gaussian noise (read noise)
    gaussian_noise = np.random.normal(0, 5, image.shape).astype(np.int16)
    noisy = cv2.add(noisy.astype(np.int16), gaussian_noise, dtype=cv2.CV_16S)
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)

    return noisy


# Optional: Add motion blur for realism
def apply_motion_blur(image, kernel_size=5):
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
    kernel = kernel / kernel_size
    return cv2.filter2D(image, -1, kernel)


# --- Video Processing Functions ---


def process_video(
    input_path, output_path, brightness_factor=20, target_res=(1280, 720)
):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error opening video: {input_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, target_res)

    for frame in frame_generator(input_path):
        # Resize frame to 720p
        frame = cv2.resize(frame, target_res)

        # Simulate reduced brightness (underexposure)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.subtract(v, brightness_factor)
        hsv = cv2.merge((h, s, v))
        frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Apply realistic degradation
        frame = add_realistic_noise(frame)
        frame = apply_motion_blur(frame)

        # cv2.imshow("Processed Frame", frame)
        # if cv2.waitKey(1) & 0xFF == ord("q"):
        #     break

        out.write(frame)

    out.release()
    cv2.destroyAllWindows()


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


# --- Run the full processing pipeline ---
process_all_videos(input_folder, output_folder)

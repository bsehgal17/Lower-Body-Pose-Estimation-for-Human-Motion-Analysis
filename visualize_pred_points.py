import cv2
import json
import numpy as np
import matplotlib.pyplot as plt


def visualize_predictions(video_path, json_file, frame_range):
    """
    Visualizes predicted keypoints from a JSON file on video frames with their indices.

    Parameters:
    - video_path (str): Path to the video file.
    - json_file (str): Path to the JSON file containing predicted keypoints.
    - frame_range (tuple): Start and end frame indices.

    Returns:
    - None (Displays video frames with overlaid keypoints).
    """

    # Load JSON file
    with open(json_file, "r") as f:
        predictions = json.load(f)

    # Open video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Couldn't open the video.")
        return

    # Define frame range
    start_frame, end_frame = frame_range
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Ensure end frame doesn't exceed video length
    end_frame = min(end_frame, total_frames)

    for frame_num in range(start_frame, end_frame):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()

        if not ret:
            print(f"Error: Couldn't read frame {frame_num}.")
            continue

        # Convert frame to RGB for plotting
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Extract keypoints for the current frame
        if frame_num >= len(predictions):
            print(f"Warning: No predictions available for frame {frame_num}")
            continue

        keypoints = np.array(predictions[frame_num]["keypoints"][0]["keypoints"][0])

        # Plot frame
        plt.figure(figsize=(8, 8))
        plt.imshow(frame_rgb)

        # Overlay keypoints with indices
        for idx, (x, y) in enumerate(keypoints):
            plt.scatter(x, y, c="r", marker="o", s=50)
            plt.text(x + 3, y, str(idx), fontsize=8, color="white")

        plt.title(f"Frame {frame_num} - Predicted Keypoints")
        plt.xlabel("X")
        plt.ylabel("Y")

        # Show the frame
        plt.show()

    # Release video capture
    cap.release()

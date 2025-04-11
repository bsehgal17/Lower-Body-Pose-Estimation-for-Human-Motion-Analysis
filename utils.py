import os
import cv2
import json
from config import VIDEO_EXTENSIONS
import matplotlib.pyplot as plt


def get_video_files(video_folder):
    """Returns a list of all video file paths in the given folder and its subfolders."""
    video_files = []
    for dirpath, _, filenames in os.walk(video_folder):
        for f in filenames:
            if f.lower().endswith(VIDEO_EXTENSIONS):
                video_files.append(os.path.join(dirpath, f))
    return video_files


def frame_generator(video_path):
    """Generator to read frames from a video file."""
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        print(f"Error: Couldn't open video {video_path}")
        return

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        yield frame

    video_capture.release()


def combine_keypoints(pose_results, frame_idx, video_data, bboxes):
    """Appends keypoints and predictions for each frame to the video_data list."""
    frame_data = {
        "frame_idx": frame_idx,
        "keypoints": [
            {
                "keypoints": person.pred_instances.keypoints.tolist(),
                "scores": person.pred_instances.keypoint_scores.tolist(),
                "bboxes": bboxes.tolist(),
            }
            for person in pose_results
        ],
    }
    video_data.append(frame_data)


def save_keypoints_to_json(video_data, output_dir, video_name):
    """Saves all keypoints and predictions for the entire video in a single JSON file."""
    video_output_dir = os.path.join(output_dir, os.path.splitext(video_name)[0])
    os.makedirs(video_output_dir, exist_ok=True)

    # Create a final output file for the entire video
    output_file = os.path.join(
        video_output_dir, f"{os.path.splitext(video_name)[0]}.json"
    )

    # Save the collected data into a single JSON file
    with open(output_file, "w") as f:
        json.dump(video_data, f, indent=4)

    print(f"Saved keypoints and predictions for the entire video to {output_file}")


def plot_filtering_effect(
    original, filtered, title="Signal Filtering Comparison", save_path=None
):
    """
    Plots original vs filtered signals on the same plot.

    Args:
        original (array): Raw signal data.
        filtered (array): Processed signal data.
        title (str): Plot title.
        save_path (str): If provided, saves plot to this path.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(original, "b-", label="Original", alpha=0.7)
    plt.plot(filtered, "r-", label="Filtered", linewidth=1)
    plt.title(title)
    plt.xlabel("Frame Number")
    plt.ylabel("Coordinate Value")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()  # Prevents display if used in batch processing

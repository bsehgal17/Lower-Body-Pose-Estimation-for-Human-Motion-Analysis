import os
import cv2
import json
from config import VIDEO_EXTENSIONS


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
                "bboxes": bboxes.tolist()
            }
            for person in pose_results
        ]
    }
    video_data.append(frame_data)


def save_keypoints_to_json(video_data, output_dir, video_name):
    """Saves all keypoints and predictions for the entire video in a single JSON file."""
    video_output_dir = os.path.join(
        output_dir, os.path.splitext(video_name)[0])
    os.makedirs(video_output_dir, exist_ok=True)

    # Create a final output file for the entire video
    output_file = os.path.join(
        video_output_dir, f"{os.path.splitext(video_name)[0]}.json")

    # Save the collected data into a single JSON file
    with open(output_file, "w") as f:
        json.dump(video_data, f, indent=4)

    print(
        f"Saved keypoints and predictions for the entire video to {output_file}")

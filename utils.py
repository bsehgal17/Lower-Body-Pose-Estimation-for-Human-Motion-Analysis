import os
import cv2
import json
from config import VIDEO_EXTENSIONS

def get_video_files(video_folder):
    """Returns a list of all video file paths in the given folder."""
    return [os.path.join(video_folder, f) for f in os.listdir(video_folder) if f.lower().endswith(VIDEO_EXTENSIONS)]


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


def save_keypoints_to_json(pose_results, frame_idx, output_dir, video_name):
    """Saves keypoints as JSON per frame in a folder named after the video."""
    video_output_dir = os.path.join(output_dir, os.path.splitext(video_name)[0])
    os.makedirs(video_output_dir, exist_ok=True)

    output_file = os.path.join(video_output_dir, f"frame_{frame_idx:04d}.json")
    keypoint_data = [
        {
            "keypoints": person.pred_instances.keypoints.tolist(),
            "scores": person.pred_instances.keypoint_scores.tolist()
        }
        for person in pose_results
    ]
    with open(output_file, "w") as f:
        json.dump(keypoint_data, f, indent=4)

    print(f"Saved keypoints of frame {frame_idx} to {output_file}")

import os
import cv2
from config.base import get_config


def get_video_files(video_folder, config=None):
    """Returns a list of all video file paths in the given folder and its subfolders."""
    if config is None:
        config = get_config()
    video_extensions = config.video.extensions
    video_files = []
    for dirpath, _, filenames in os.walk(video_folder):
        for f in filenames:
            if f.lower().endswith(video_extensions):
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

import os
import cv2
import logging
from typing import List
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


def get_video_files(video_folder: str, video_exts: List[str]) -> List[str]:
    """
    Returns a list of all video file paths in the given folder and its subfolders
    matching the provided video extensions.
    """
    video_files = []
    for dirpath, _, filenames in os.walk(video_folder):
        for f in filenames:
            if any(f.lower().endswith(ext) for ext in video_exts):
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


def get_video_resolution(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return width, height


def rescale_keypoints(pred_keypoints, scale_x, scale_y):
    rescaled = []
    for frame in pred_keypoints:
        rescaled_frame = [[x * scale_x, y * scale_y] for x, y in frame]
        rescaled.append(rescaled_frame)
    return rescaled


def extract_frames_from_video_tree(
    input_root: str,
    output_root: str,
    supported_exts: Tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv"),
    resize: Optional[Tuple[int, int]] = None,
    max_frames: Optional[int] = None,
    frame_skip: int = 1,
):
    """
    Recursively extract frames from videos in a directory tree and save with mirrored structure.

    Args:
        input_root (str): Root directory containing videos.
        output_root (str): Output directory to save extracted frames.
        supported_exts (Tuple[str, ...]): File extensions to consider as video.
        resize (Optional[Tuple[int, int]]): Resize (width, height) if provided.
        max_frames (Optional[int]): Max frames per video.
        frame_skip (int): Save every Nth frame.
    """
    for dirpath, _, filenames in os.walk(input_root):
        for filename in filenames:
            if filename.lower().endswith(supported_exts):
                video_path = os.path.join(dirpath, filename)
                relative_path = os.path.relpath(dirpath, input_root)
                output_folder = os.path.join(
                    output_root, relative_path, os.path.splitext(filename)[0]
                )
                try:
                    cap = cv2.VideoCapture(video_path)
                    if not cap.isOpened():
                        print(f"Failed to open video: {video_path}")
                        continue

                    os.makedirs(output_folder, exist_ok=True)

                    frame_count = 0
                    saved_count = 0
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        if frame_count % frame_skip == 0:
                            if resize:
                                frame = cv2.resize(frame, resize)
                            frame_name = f"frame_{saved_count:05d}.jpg"
                            cv2.imwrite(os.path.join(output_folder, frame_name), frame)
                            saved_count += 1
                            if max_frames and saved_count >= max_frames:
                                break
                        frame_count += 1

                    cap.release()
                    print(
                        f"Extracted {saved_count} frames from {video_path} to {output_folder}"
                    )
                except Exception as e:
                    print(f"Error processing {video_path}: {e}")

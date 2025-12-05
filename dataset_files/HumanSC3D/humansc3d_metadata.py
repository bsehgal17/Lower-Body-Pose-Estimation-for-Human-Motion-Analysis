import os
import re
from typing import Dict


def get_humansc3d_metadata_from_json(video_path: str) -> Dict:
    """
    Extracts metadata for HumanSC3D dataset from a given video path.

    Expected path format:
    C:/Users/BhavyaSehgal/Downloads/humansc3d/humansc3d_train/train/s06/videos/50591643/170.mp4

    Where:
    - Subject folder like 's06', 's01', 's02', etc.
    - Camera folder like '50591643', '58860488', etc.
    - Filename like '170.mp4', '001.mp4', etc.

    Returns:
        Dict with keys: 'subject', 'action', 'camera'
    """
    norm_path = os.path.normpath(video_path)
    parts = norm_path.split(os.sep)

    # Extract subject like "s01", "s02", "s06" from the path
    subject = next((p for p in parts if re.fullmatch(r"s\d+", p)), None)

    # Extract camera ID from parent directory of the video file
    camera = None
    if len(parts) >= 3:
        camera = parts[-3]  # The directory containing the video file

    # Extract action from filename (without extension)
    action = os.path.splitext(os.path.basename(video_path))[0]

    if not subject or not camera or not action:
        raise ValueError(
            f"Could not extract HumanSC3D metadata from: {video_path}"
        )

    return {"subject": subject, "action": action, "camera": camera}


def get_humansc3d_metadata_from_video(video_path: str) -> Dict:
    """
    Extracts metadata for HumanSC3D dataset from a given video path.

    Only processes real video files.
    Allowed extensions: .mp4, .avi, .mov, .mkv
    """

    # -------- 1. Check extension --------
    valid_exts = {".mp4", ".avi", ".mov", ".mkv"}
    ext = os.path.splitext(video_path)[1].lower()

    if ext not in valid_exts:
        raise ValueError(f"Not a video file: {video_path}")

    # -------- 2. Normalize and split path --------
    norm_path = os.path.normpath(video_path)
    parts = norm_path.split(os.sep)

    # -------- 3. Extract subject (s01, s02, s06, ...) --------
    subject = next((p for p in parts if re.fullmatch(r"s\d+", p)), None)

    # -------- 4. Extract camera (the parent folder of video file) --------
    camera = parts[-2] if len(parts) >= 2 else None

    # -------- 5. Extract action (filename without extension) --------
    action = os.path.splitext(os.path.basename(video_path))[0]

    # -------- 6. Validate values --------
    if not subject:
        raise ValueError(f"Could not detect subject in path: {video_path}")

    if not camera:
        raise ValueError(f"Could not detect camera in path: {video_path}")

    if not action.isdigit():
        # If actions are always numbers like 170, 001, treat others as invalid
        raise ValueError(f"Invalid action filename: {video_path}")

    return {"subject": subject, "action": action, "camera": camera}


def get_humansc3d_gt_path(video_path: str, gt_root: str = None) -> str:
    """
    Generate ground truth path for HumanSC3D dataset from video path.

    Args:
        video_path: Path to video file
        gt_root: Optional ground truth root directory. If None, uses processed_outputs in same subject folder

    Returns:
        Path to corresponding ground truth JSON file
    """
    metadata = get_humansc3d_metadata_from_video(video_path)

    if gt_root is None:
        # Use processed_outputs in the same subject folder
        norm_path = os.path.normpath(video_path)
        parts = norm_path.split(os.sep)

        # Find the subject folder index
        subject_idx = None
        for i, part in enumerate(parts):
            if re.fullmatch(r"s\d+", part):
                subject_idx = i
                break

        if subject_idx is None:
            raise ValueError(
                f"Cannot find subject folder in path: {video_path}")

        # Build path correctly
        gt_root = os.sep.join(
            parts[:subject_idx+1] + ["processed_outputs", "2d_points"]
        )

    # Generate GT filename: {action}_{camera}_2d.json
    gt_filename = f"{metadata['action']}_{metadata['camera']}_2d.json"
    gt_path = os.path.join(gt_root, gt_filename)

    return gt_path

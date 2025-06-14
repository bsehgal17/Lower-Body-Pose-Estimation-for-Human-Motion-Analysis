import os
import re
from typing import Dict


def get_humaneva_metadata_from_video(video_path: str) -> Dict:
    """
    Extracts metadata for HumanEva dataset from a given video path.

    Assumes path contains:
    - Subject folder like 'S1', 'S2', 'S3'
    - Filename like 'Walking_1_(C2).avi'

    Returns:
        Dict with keys: 'subject', 'action', 'camera'
    """
    norm_path = os.path.normpath(video_path)
    parts = norm_path.split(os.sep)

    # Extract subject like "S1", "S2", "S3" from the path
    subject = next((p for p in parts if re.fullmatch(r"S\d+", p)), None)

    # Extract filename stem
    file_stem = os.path.splitext(os.path.basename(video_path))[
        0
    ]  # e.g., Walking_1_(C2)

    # Parse action and camera from filename
    match = re.match(r"(.+)_\(C(\d+)\)", file_stem)
    if not subject or not match:
        raise ValueError(f"Could not extract HumanEva metadata from: {video_path}")

    action = match.group(1).replace("_", " ")
    camera = f"C{match.group(2)}"

    return {"subject": subject, "action": action, "camera": camera}

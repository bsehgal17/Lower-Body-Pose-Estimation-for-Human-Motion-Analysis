import re


def parse_movi_metadata(filename: str):
    """
    Expected format: Subject_<id>_segment_<segment>_cam_<camera>.pkl
    Example: Subject_3_segment_12_cam_2.pkl
    """
    match = re.match(r"Subject_(\d+)_segment_(\d+)_cam_(\d+)", filename)
    if not match:
        raise ValueError(
            f"Filename '{filename}' does not match expected pattern.")
    return int(match.group(1)), int(match.group(2)), int(match.group(3))

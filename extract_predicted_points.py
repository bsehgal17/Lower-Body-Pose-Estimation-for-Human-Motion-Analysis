import numpy as np
import json


def extract_predictions(json_file, frame_range=None):
    """
    Extracts keypoint predictions from a JSON file.
    If frame_range is provided, extracts keypoints within the specified range.
    If frame_range is None, extracts keypoints for all frames.

    Parameters:
    - json_file: Path to the JSON file containing keypoint predictions.
    - frame_range: Tuple (start_frame, end_frame) specifying the range of frames to extract.
                   If None, extracts all frames.

    Returns:
    - pred_keypoints: A NumPy array of extracted keypoints.
    """
    print("Loading JSON file (Predictions)...")
    with open(json_file, "r") as f:
        json_data = json.load(f)
    print(f"Loaded JSON file with {len(json_data)} frames")

    # Determine frame range
    if frame_range is None:
        frame_range = (0, len(json_data))

    # Extract all available keypoints
    keypoint_indices = {
        f"keypoint_{i}": i
        for i in range(len(json_data[0]["keypoints"][0]["keypoints"][0]))
    }

    pred_keypoints = []
    frame_end = min(frame_range[1], len(json_data))

    for i in range(frame_range[0], frame_end):
        pred_frame = []
        json_kpts = np.array(json_data[i]["keypoints"][0]["keypoints"][0])

        for _, index in keypoint_indices.items():
            pred_point = json_kpts[index]
            pred_frame.append(pred_point)

        pred_keypoints.append(np.array(pred_frame))

    return np.array(pred_keypoints)

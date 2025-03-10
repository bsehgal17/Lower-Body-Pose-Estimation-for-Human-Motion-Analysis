import numpy as np
import json


def extract_predictions(json_file, frame_range):
    """
    Extracts keypoint predictions from a JSON file within a specified frame range.
    Allows selecting specific keypoints, or extracts all if none are specified.

    Parameters:
    - json_file: Path to the JSON file containing keypoint predictions.
    - frame_range: Tuple (start_frame, end_frame) specifying the range of frames to extract.
    - keypoint_indices: Dictionary mapping keypoint names to indices (e.g., {'left_hip': 11, 'right_hip': 12}).
                        If None, extracts all available keypoints.

    Returns:
    - pred_keypoints: A NumPy array of extracted keypoints within the given frame range.
    """
    print("Loading JSON file (Predictions)...")
    with open(json_file, "r") as f:
        json_data = json.load(f)
    print(f"Loaded JSON file with {len(json_data)} frames")

    # If no specific keypoints are provided, extract all available keypoints
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

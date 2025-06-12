import json
import os
import numpy as np
from typing import List, Dict, Any
from utils.video_info import extract_video_info
from config import base
from utils.joint_enum import PredJoints


def euclidean_distance(point1: List[float], point2: List[float]) -> float:
    """
    Computes the Euclidean distance between two 2D points.

    Args:
        point1: [x, y] coordinates of the first point.
        point2: [x, y] coordinates of the second point.

    Returns:
        Euclidean distance as a float.
    """
    return np.linalg.norm(np.array(point1) - np.array(point2))


def correct_keypoints(
    keypoints: List[Dict[str, Any]], threshold: float
) -> List[Dict[str, Any]]:
    """
    Corrects outlier keypoints in a sequence of frames by replacing them with
    the average of spatial neighbors if their distances from adjacent frames are too large.

    Args:
        keypoints: List of frame-wise keypoint dictionaries (typically parsed from JSON).
        threshold: Distance threshold beyond which a keypoint is considered an outlier.

    Returns:
        A corrected version of the keypoints with smoothed replacements where necessary.
    """
    num_frames = len(keypoints)
    corrected_keypoints = keypoints.copy()

    for frame_idx in range(num_frames):
        for keypoint_group in keypoints[frame_idx]["keypoints"]:
            for keypoint_set_idx in range(len(keypoint_group["keypoints"])):
                for joint_idx in lower_body_joints:
                    current_point = keypoint_group["keypoints"][keypoint_set_idx][
                        joint_idx
                    ]

                    # Get previous and next points for distance check
                    prev_point = (
                        keypoints[frame_idx - 1]["keypoints"][0]["keypoints"][
                            keypoint_set_idx
                        ][joint_idx]
                        if frame_idx > 0
                        else current_point
                    )
                    next_point = (
                        keypoints[frame_idx + 1]["keypoints"][0]["keypoints"][
                            keypoint_set_idx
                        ][joint_idx]
                        if frame_idx < num_frames - 1
                        else current_point
                    )

                    dist_prev = euclidean_distance(current_point, prev_point)
                    dist_next = euclidean_distance(current_point, next_point)

                    # Replace if both distances exceed threshold
                    if dist_prev > threshold and dist_next > threshold:
                        prev_points: List[List[float]] = []
                        next_points: List[List[float]] = []

                        for i in range(1, 4):
                            if frame_idx - i >= 0:
                                prev_point = keypoints[frame_idx - i]["keypoints"][0][
                                    "keypoints"
                                ][keypoint_set_idx][joint_idx]
                                prev_points.append(prev_point)

                        for i in range(1, 4):
                            if frame_idx + i < num_frames:
                                next_point = keypoints[frame_idx + i]["keypoints"][0][
                                    "keypoints"
                                ][keypoint_set_idx][joint_idx]
                                next_points.append(next_point)

                        all_points = prev_points + next_points

                        if len(all_points) > 0:
                            avg_x = sum(p[0] for p in all_points) / len(all_points)
                            avg_y = sum(p[1] for p in all_points) / len(all_points)
                            corrected_keypoints[frame_idx]["keypoints"][0]["keypoints"][
                                keypoint_set_idx
                            ][joint_idx] = [float(avg_x), float(avg_y)]

    return corrected_keypoints


def save_corrected_keypoints(
    output_folder: str,
    original_json_path: str,
    corrected_keypoints: List[Dict[str, Any]],
) -> None:
    """
    Saves the corrected keypoints to disk as a new JSON file.

    Args:
        output_folder: Directory where the corrected file will be saved.
        original_json_path: Path to the original JSON file (used to derive filename).
        corrected_keypoints: List of updated keypoints to save.
    """
    os.makedirs(output_folder, exist_ok=True)
    corrected_json_path = os.path.join(
        output_folder,
        os.path.basename(original_json_path).replace(
            ".json", "_neighbor_corrected.json"
        ),
    )
    with open(corrected_json_path, "w") as f:
        json.dump(corrected_keypoints, f, indent=4)
    print(f"Corrected keypoints saved to: {corrected_json_path}")


# === Main Execution ===
base_path: str = base.VIDEO_FOLDER
threshold: float = 5.0

lower_body_joints: List[int] = [
    PredJoints.LEFT_ANKLE.value,
    PredJoints.RIGHT_ANKLE.value,
    PredJoints.LEFT_HIP.value,
    PredJoints.RIGHT_HIP.value,
    PredJoints.LEFT_KNEE.value,
    PredJoints.RIGHT_KNEE.value,
]

output_base: str = (
    r"C:\Users\BhavyaSehgal\Downloads\bhavya_1st_sem\humaneva\rtmw_x_degraded_40"
)

for root, dirs, files in os.walk(base_path):
    for file in files:
        video_info = extract_video_info(file, root)
        if video_info:
            subject, action, camera = video_info
            action_group = action.replace(" ", "_")
            json_path = os.path.join(
                output_base,
                subject,
                f"{action_group}_({'C' + str(camera + 1)})",
                "gaussian",
                f"{action_group}_({'C' + str(camera + 1)})".replace(" ", "")
                + "_gaussian_filtered.json",
            )

            if not os.path.exists(json_path):
                print(f"File not found: {json_path}")
                continue

            with open(json_path, "r") as f:
                pred_keypoints: List[Dict[str, Any]] = json.load(f)

            corrected_keypoints = correct_keypoints(pred_keypoints, threshold)

            output_folder = os.path.join(
                output_base,
                subject,
                f"{action_group}_({'C' + str(camera + 1)})",
                "neighbor_corrected",
            )
            save_corrected_keypoints(output_folder, json_path, corrected_keypoints)

print("Processing complete.")

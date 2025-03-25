import json
import os
import numpy as np
from video_info import extract_video_info
import config
from joint_enum import PredJoints


def moving_average_filter(keypoints, window_size=3):
    """Applies a moving average filter to smooth keypoints over time."""
    num_frames = len(keypoints)
    half_window = window_size // 2
    smoothed_keypoints = keypoints.copy()

    for frame_idx in range(num_frames):
        for keypoint_group in keypoints[frame_idx]["keypoints"]:
            for keypoint_set in keypoint_group["keypoints"]:
                for joint_idx in lower_body_joints:
                    x_vals, y_vals = [], []

                    # Collect values from neighboring frames
                    for offset in range(-half_window, half_window + 1):
                        neighbor_idx = max(0, min(frame_idx + offset, num_frames - 1))
                        neighbor_point = keypoints[neighbor_idx]["keypoints"][0][
                            "keypoints"
                        ][joint_idx]
                        x_vals.append(neighbor_point[0])
                        y_vals.append(neighbor_point[1])

                    # Compute moving average
                    smoothed_x = np.mean(x_vals)
                    smoothed_y = np.mean(y_vals)

                    smoothed_keypoints[frame_idx]["keypoints"][0]["keypoints"][
                        joint_idx
                    ] = [smoothed_x, smoothed_y]

    return smoothed_keypoints


def save_smoothed_keypoints(original_json_path, smoothed_keypoints):
    smoothed_json_path = original_json_path.replace(".json", "_smoothed.json")
    with open(smoothed_json_path, "w") as f:
        json.dump(smoothed_keypoints, f, indent=4)
    print(f"Smoothed keypoints saved to: {smoothed_json_path}")


base_path = config.VIDEO_FOLDER
window_size = 5  # Set the window size for the moving average filter
lower_body_joints = [
    PredJoints.LEFT_ANKLE.value,
    PredJoints.RIGHT_ANKLE.value,
    PredJoints.LEFT_HIP.value,
    PredJoints.RIGHT_HIP.value,
    PredJoints.LEFT_KNEE.value,
    PredJoints.RIGHT_KNEE.value,
]

for root, dirs, files in os.walk(base_path):
    for file in files:
        video_info = extract_video_info(file, root)
        if video_info:
            subject, action, camera = video_info
            action_group = action.replace(" ", "_")
            json_path = os.path.join(
                r"C:\Users\BhavyaSehgal\Downloads\bhavya_1st_sem\humaneva\HumanEva\rtmw_x_degraded",
                subject,
                f"{action_group}_({'C' + str(camera + 1)})",
                f"{action_group}_({'C' + str(camera + 1)})/{action_group}_({'C' + str(camera + 1)})".replace(
                    " ", ""
                )
                + ".json",
            )

            with open(json_path, "r") as f:
                pred_keypoints = json.load(f)

            smoothed_keypoints = moving_average_filter(pred_keypoints, window_size)
            save_smoothed_keypoints(json_path, smoothed_keypoints)

print("Processing complete.")

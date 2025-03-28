import json
import os
import numpy as np
from video_info import extract_video_info
import config
from joint_enum import PredJoints


def moving_average_filter(data, window_size=5):
    """Applies a moving average filter for temporal smoothing."""
    num_frames = len(data)
    if num_frames < window_size:
        return data  # Not enough data for filtering

    half_window = window_size // 2
    smoothed_data = np.copy(data)

    for i in range(num_frames):
        start_idx = max(0, i - half_window)
        end_idx = min(num_frames, i + half_window + 1)
        smoothed_data[i] = np.mean(data[start_idx:end_idx], axis=0)

    return smoothed_data


def save_filtered_keypoints(output_folder, original_json_path, filtered_keypoints):
    os.makedirs(output_folder, exist_ok=True)
    filtered_json_path = os.path.join(
        output_folder,
        os.path.basename(original_json_path).replace(
            ".json", "_moving_avg_filtered.json"
        ),
    )
    with open(filtered_json_path, "w") as f:
        json.dump(filtered_keypoints, f, indent=4)
    print(f"Filtered keypoints saved to: {filtered_json_path}")


base_path = config.VIDEO_FOLDER
window_size = 5
lower_body_joints = [
    PredJoints.LEFT_ANKLE.value,
    PredJoints.RIGHT_ANKLE.value,
    PredJoints.LEFT_HIP.value,
    PredJoints.RIGHT_HIP.value,
    PredJoints.LEFT_KNEE.value,
    PredJoints.RIGHT_KNEE.value,
]

output_base = (
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
                f"{action_group}_({'C' + str(camera + 1)})/{action_group}_({'C' + str(camera + 1)})".replace(
                    " ", ""
                )
                + ".json",
            )

            if not os.path.exists(json_path):
                print(f"File not found: {json_path}")
                continue

            with open(json_path, "r") as f:
                pred_keypoints = json.load(f)

            for keypoint_set_idx in range(
                len(pred_keypoints[0]["keypoints"][0]["keypoints"])
            ):
                for joint_idx in lower_body_joints:
                    x_series = []
                    y_series = []

                    for frame_data in pred_keypoints:
                        kp = frame_data["keypoints"][0]["keypoints"][keypoint_set_idx][
                            joint_idx
                        ]
                        x_series.append(kp[0])
                        y_series.append(kp[1])

                    x_filtered = moving_average_filter(np.array(x_series), window_size)
                    y_filtered = moving_average_filter(np.array(y_series), window_size)

                    for i, frame_data in enumerate(pred_keypoints):
                        frame_data["keypoints"][0]["keypoints"][keypoint_set_idx][
                            joint_idx
                        ][0] = float(x_filtered[i])
                        frame_data["keypoints"][0]["keypoints"][keypoint_set_idx][
                            joint_idx
                        ][1] = float(y_filtered[i])

            output_folder = os.path.join(
                output_base,
                subject,
                f"{action_group}_({'C' + str(camera + 1)})",
                "moving_average",
            )
            save_filtered_keypoints(output_folder, json_path, pred_keypoints)

print("Moving average filtering complete.")

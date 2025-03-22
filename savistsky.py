import json
import os
import scipy.signal
from video_info import extract_video_info
import config
from joint_enum import PredJoints


def save_filtered_keypoints(original_json_path, filtered_keypoints):
    filtered_json_path = original_json_path.replace(".json", "_savgol_filtered.json")
    with open(filtered_json_path, "w") as f:
        json.dump(filtered_keypoints, f, indent=4)
    print(f"Filtered keypoints saved to: {filtered_json_path}")


base_path = config.VIDEO_FOLDER
window_size = 5  # Set an odd window size for the Savitzky-Golay filter
polyorder = 2  # Set polynomial order for the filter
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

            for frame_data in pred_keypoints:
                for keypoint_group in frame_data["keypoints"]:
                    for keypoint_set in keypoint_group["keypoints"]:
                        for joint_idx in lower_body_joints:
                            joint_x_series = [
                                keypoint_set[i][0]
                                for i in range(
                                    max(0, joint_idx - 3),
                                    min(len(keypoint_set), joint_idx + 4),
                                )
                            ]
                            joint_y_series = [
                                keypoint_set[i][1]
                                for i in range(
                                    max(0, joint_idx - 3),
                                    min(len(keypoint_set), joint_idx + 4),
                                )
                            ]
                            smoothed_x = scipy.signal.savgol_filter(
                                joint_x_series, window_size, polyorder
                            )
                            smoothed_y = scipy.signal.savgol_filter(
                                joint_y_series, window_size, polyorder
                            )

                            keypoint_set[joint_idx] = [
                                float(smoothed_x[len(smoothed_x) // 2]),
                                float(smoothed_y[len(smoothed_y) // 2]),
                            ]

            save_filtered_keypoints(json_path, pred_keypoints)

print("Processing complete.")

import json
import os
import scipy.ndimage
from utils.video_info import extract_video_info
import config
from utils.joint_enum import PredJoints


def save_filtered_keypoints(output_folder, original_json_path, filtered_keypoints):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    filtered_json_path = os.path.join(
        output_folder,
        os.path.basename(original_json_path).replace(
            ".json", "_butter_median_filtered.json"
        ),
    )
    with open(filtered_json_path, "w") as f:
        json.dump(filtered_keypoints, f, indent=4)
    print(f"Filtered keypoints saved to: {filtered_json_path}")


base_path = config.VIDEO_FOLDER
window_size = 11  # You can adjust this to tune smoothing (must be odd integer)
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
                "butterworth",
                f"{action_group}_({'C' + str(camera + 1)})".replace(" ", "")
                + "_butter_filtered.json",
            )

            if not os.path.exists(json_path):
                print(f"File not found: {json_path}")
                continue

            with open(json_path, "r") as f:
                pred_keypoints = json.load(f)

                # Temporal smoothing of each joint's x and y coordinates
                for keypoint_set_idx in range(
                    len(pred_keypoints[0]["keypoints"][0]["keypoints"])
                ):
                    for joint_idx in lower_body_joints:
                        x_series = []
                        y_series = []
                        for frame_data in pred_keypoints:
                            kp = frame_data["keypoints"][0]["keypoints"][
                                keypoint_set_idx
                            ][joint_idx]
                            x_series.append(kp[0])
                            y_series.append(kp[1])

                        # Apply median filtering
                        smoothed_x = scipy.ndimage.median_filter(
                            x_series, size=window_size
                        )
                        smoothed_y = scipy.ndimage.median_filter(
                            y_series, size=window_size
                        )

                        for i, frame_data in enumerate(pred_keypoints):
                            frame_data["keypoints"][0]["keypoints"][keypoint_set_idx][
                                joint_idx
                            ][0] = float(smoothed_x[i])
                            frame_data["keypoints"][0]["keypoints"][keypoint_set_idx][
                                joint_idx
                            ][1] = float(smoothed_y[i])

            output_folder = os.path.join(
                output_base,
                subject,
                f"{action_group}_({'C' + str(camera + 1)})",
                "median",  # Changed folder name to reflect median filtering
            )
            save_filtered_keypoints(output_folder, json_path, pred_keypoints)

print("Processing complete.")

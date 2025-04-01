import json
import os
import numpy as np
from video_info import extract_video_info
import config
from joint_enum import PredJoints


def euclidean_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))


def correct_keypoints(keypoints, threshold):
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

                    if dist_prev > threshold and dist_next > threshold:
                        # Collect points from previous 3 frames and next 3 frames
                        prev_points = []
                        next_points = []

                        # Get previous frames (up to 3)
                        for i in range(1, 4):
                            if frame_idx - i >= 0:
                                prev_point = keypoints[frame_idx - i]["keypoints"][0][
                                    "keypoints"
                                ][keypoint_set_idx][joint_idx]
                                prev_points.append(prev_point)

                        # Get next frames (up to 3)
                        for i in range(1, 4):
                            if frame_idx + i < num_frames:
                                next_point = keypoints[frame_idx + i]["keypoints"][0][
                                    "keypoints"
                                ][keypoint_set_idx][joint_idx]
                                next_points.append(next_point)

                        # Combine all points (prev + next)
                        all_points = prev_points + next_points

                        if len(all_points) > 0:
                            # Calculate mean of all available points
                            avg_x = sum(p[0] for p in all_points) / len(all_points)
                            avg_y = sum(p[1] for p in all_points) / len(all_points)
                            corrected_keypoints[frame_idx]["keypoints"][0]["keypoints"][
                                keypoint_set_idx
                            ][joint_idx] = [float(avg_x), float(avg_y)]

    return corrected_keypoints


def save_corrected_keypoints(output_folder, original_json_path, corrected_keypoints):
    os.makedirs(output_folder, exist_ok=True)  # Ensure the output folder exists
    corrected_json_path = os.path.join(
        output_folder,
        os.path.basename(original_json_path).replace(
            ".json", "_neighbor_corrected.json"
        ),
    )
    with open(corrected_json_path, "w") as f:
        json.dump(corrected_keypoints, f, indent=4)
    print(f"Corrected keypoints saved to: {corrected_json_path}")


base_path = config.VIDEO_FOLDER
threshold = 5  # Set the threshold distance for correction
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
                "gaussian",
                f"{action_group}_({'C' + str(camera + 1)})".replace(" ", "")
                + "_gaussian_filtered.json",
            )

            if not os.path.exists(json_path):
                print(f"File not found: {json_path}")
                continue

            with open(json_path, "r") as f:
                pred_keypoints = json.load(f)

            corrected_keypoints = correct_keypoints(pred_keypoints, threshold)

            output_folder = os.path.join(
                output_base,
                subject,
                f"{action_group}_({'C' + str(camera + 1)})",
                "neighbor_corrected",
            )
            save_corrected_keypoints(output_folder, json_path, corrected_keypoints)

print("Processing complete.")

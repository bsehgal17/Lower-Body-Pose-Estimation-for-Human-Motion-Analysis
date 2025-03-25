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
            for keypoint_set in keypoint_group["keypoints"]:
                for joint_idx in lower_body_joints:
                    current_point = keypoint_set[joint_idx]
                    prev_point = (
                        keypoints[frame_idx - 1]["keypoints"][0]["keypoints"][joint_idx]
                        if frame_idx > 0
                        else current_point
                    )
                    next_point = (
                        keypoints[frame_idx + 1]["keypoints"][0]["keypoints"][joint_idx]
                        if frame_idx < num_frames - 1
                        else current_point
                    )

                    dist_prev = euclidean_distance(current_point, prev_point)
                    dist_next = euclidean_distance(current_point, next_point)

                    if dist_prev > threshold and dist_next > threshold:
                        avg_x = (prev_point[0] + next_point[0]) / 2
                        avg_y = (prev_point[1] + next_point[1]) / 2
                        corrected_keypoints[frame_idx]["keypoints"][0]["keypoints"][
                            joint_idx
                        ] = [avg_x, avg_y]

    return corrected_keypoints


def save_corrected_keypoints(original_json_path, corrected_keypoints):
    corrected_json_path = original_json_path.replace(".json", "_corrected.json")
    with open(corrected_json_path, "w") as f:
        json.dump(corrected_keypoints, f, indent=4)
    print(f"Corrected keypoints saved to: {corrected_json_path}")


base_path = config.VIDEO_FOLDER
threshold = 50  # Set the threshold distance for correction
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

            corrected_keypoints = correct_keypoints(pred_keypoints, threshold)
            save_corrected_keypoints(json_path, corrected_keypoints)

print("Processing complete.")

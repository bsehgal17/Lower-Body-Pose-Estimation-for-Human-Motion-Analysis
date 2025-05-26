import json
import os
import numpy as np
from pykalman import KalmanFilter
from utils.video_info import extract_video_info
from utils import config

from utils.joint_enum import PredJoints


def save_filtered_keypoints(output_folder, original_json_path, filtered_keypoints):
    os.makedirs(output_folder, exist_ok=True)
    filtered_json_path = os.path.join(
        output_folder,
        os.path.basename(original_json_path).replace(
            ".json", "_kalman_rts_smoothed.json"
        ),
    )
    with open(filtered_json_path, "w") as f:
        json.dump(filtered_keypoints, f, indent=4)
    print(f"RTS-smoothed keypoints saved to: {filtered_json_path}")


def initialize_kalman_filter(process_noise=0.01, measurement_noise=1.0):
    """Initialize a Kalman Filter with constant velocity model."""
    return KalmanFilter(
        transition_matrices=np.array([[1, 1], [0, 1]]),  # State transition
        observation_matrices=np.array([[1, 0]]),  # Observation matrix
        initial_state_mean=[0, 0],  # Initial state
        initial_state_covariance=np.eye(2),  # Initial covariance
        transition_covariance=process_noise * np.eye(2),  # Process noise
        observation_covariance=measurement_noise,  # Measurement noise
    )


def apply_kalman_rts(measurements, process_noise, measurement_noise):
    """Apply forward Kalman filter + backward RTS smoother."""
    kf = initialize_kalman_filter(process_noise, measurement_noise)

    # Forward pass (standard Kalman filter)
    filtered_state_means, filtered_state_covariances = kf.filter(measurements)

    # Backward pass (RTS smoother)
    smoothed_state_means, _ = kf.smooth(measurements)

    return smoothed_state_means[:, 0]  # Return only position estimates


# Configuration
base_path = config.VIDEO_FOLDER
output_base = (
    r"C:\Users\BhavyaSehgal\Downloads\bhavya_1st_sem\humaneva\rtmw_x_degraded_40"
)

# Tune these parameters:
PROCESS_NOISE = 5  # Higher = more responsive to changes
MEASUREMENT_NOISE = 2.0  # Higher = more aggressive smoothing

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

                for keypoint_set_idx in range(
                    len(pred_keypoints[0]["keypoints"][0]["keypoints"])
                ):
                    for joint_idx in lower_body_joints:
                        # Extract x and y coordinates
                        measurements = np.array(
                            [
                                [
                                    frame_data["keypoints"][0]["keypoints"][
                                        keypoint_set_idx
                                    ][joint_idx][0],
                                    frame_data["keypoints"][0]["keypoints"][
                                        keypoint_set_idx
                                    ][joint_idx][1],
                                ]
                                for frame_data in pred_keypoints
                            ]
                        )

                        # Apply RTS smoothing to x and y coordinates
                        smoothed_x = apply_kalman_rts(
                            measurements[:, 0],
                            process_noise=PROCESS_NOISE,
                            measurement_noise=MEASUREMENT_NOISE,
                        )
                        smoothed_y = apply_kalman_rts(
                            measurements[:, 1],
                            process_noise=PROCESS_NOISE,
                            measurement_noise=MEASUREMENT_NOISE,
                        )

                        # Update keypoints
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
                "kalman_rts",  # Changed from "kalman" to "kalman_rts"
            )
            save_filtered_keypoints(output_folder, json_path, pred_keypoints)

print("RTS smoothing complete.")

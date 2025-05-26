import json
import os
import numpy as np
from pykalman import KalmanFilter
from utils.video_info import extract_video_info
import config
from utils.joint_enum import PredJoints
from pre_process_utils import (
    remove_outliers_iqr,
    interpolate_missing_values,
)
from utils.utils import plot_filtering_effect


def save_filtered_keypoints(output_folder, original_json_path, filtered_keypoints):
    os.makedirs(output_folder, exist_ok=True)
    filtered_json_path = os.path.join(
        output_folder,
        os.path.basename(original_json_path).replace(
            ".json", "_kalman_filtered.json"),
    )
    with open(filtered_json_path, "w") as f:
        json.dump(filtered_keypoints, f, indent=4)
    print(f"Kalman-filtered keypoints saved to: {filtered_json_path}")


def initialize_kalman_filter(process_noise=0.01, measurement_noise=1.0):
    """Initialize a Kalman Filter with constant velocity model."""
    kf = KalmanFilter(
        transition_matrices=np.array([[1, 1], [0, 1]]),  # State transition
        observation_matrices=np.array([[1, 0]]),  # Observation matrix
        initial_state_mean=[0, 0],  # Initial state
        initial_state_covariance=np.eye(2),  # Initial covariance
        transition_covariance=process_noise * np.eye(2),  # Process noise
        observation_covariance=measurement_noise,  # Measurement noise
    )
    return kf


base_path = config.VIDEO_FOLDER
output_base = (
    r"C:\Users\BhavyaSehgal\Downloads\bhavya_1st_sem\humaneva\rtmw_x_degraded_40"
)

# Tune these parameters:
PROCESS_NOISE = 5  # Higher = less lag
MEASUREMENT_NOISE = 2.0  # More aggressive smoothing

lower_body_joints = [
    PredJoints.LEFT_ANKLE.value,
    PredJoints.RIGHT_ANKLE.value,
    PredJoints.LEFT_HIP.value,
    PredJoints.RIGHT_HIP.value,
    PredJoints.LEFT_KNEE.value,
    PredJoints.RIGHT_KNEE.value,
]
iqr_multiplier = 1.5
interpolation_kind = "linear"
for root, dirs, files in os.walk(base_path):
    for file in files:
        video_info = extract_video_info(file, root)
        if video_info:
            subject, action, camera = video_info
            action_group = action.replace(" ", "_")
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
                            kp = frame_data["keypoints"][0]["keypoints"][
                                keypoint_set_idx
                            ][joint_idx]
                            x_series.append(kp[0])
                            y_series.append(kp[1])

                            # Step 1: Outlier removal
                            x_cleaned = remove_outliers_iqr(
                                x_series, iqr_multiplier)
                            y_cleaned = remove_outliers_iqr(
                                y_series, iqr_multiplier)

                            # Step 2: Interpolation
                            x_interpolated = interpolate_missing_values(
                                x_cleaned, kind=interpolation_kind
                            )
                            y_interpolated = interpolate_missing_values(
                                y_cleaned, kind=interpolation_kind
                            )

                        # Extract x and y coordinates for the joint
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

                        kf_x = initialize_kalman_filter(
                            process_noise=PROCESS_NOISE,
                            measurement_noise=MEASUREMENT_NOISE,
                        )
                        kf_y = initialize_kalman_filter(
                            process_noise=PROCESS_NOISE,
                            measurement_noise=MEASUREMENT_NOISE,
                        )

                        # Filter x and y coordinates
                        # Inside your joint-filtering loop:
                        smoothed_x, _ = kf_x.filter(
                            measurements[:, 0]
                        )  # shape: (n_frames, 2)
                        smoothed_y, _ = kf_y.filter(measurements[:, 1])
                        if (
                            joint_idx == PredJoints.LEFT_ANKLE.value
                            and keypoint_set_idx == 0
                        ):
                            plot_dir = os.path.join(
                                output_base,
                                subject,
                                f"{action_group}_({'C' + str(camera + 1)})",
                                "plots",
                            )
                            os.makedirs(plot_dir, exist_ok=True)

                            # Plot X coordinates
                            plot_filtering_effect(
                                original=x_series,
                                filtered=smoothed_x[:, 0],
                                title=f"X-Coordinate: {subject} {action} (Joint {joint_idx})",
                                save_path=os.path.join(
                                    plot_dir,
                                    f"x_coord_joint_{joint_idx}_kalman.png",
                                ),
                            )

                            # Plot Y coordinates
                            plot_filtering_effect(
                                original=y_series,
                                filtered=smoothed_y[:, 0],
                                title=f"Y-Coordinate: {subject} {action} (Joint {joint_idx})",
                                save_path=os.path.join(
                                    plot_dir,
                                    f"y_coord_joint_{joint_idx}_kalman.png",
                                ),
                            )

                        for i, frame_data in enumerate(pred_keypoints):
                            frame_data["keypoints"][0]["keypoints"][keypoint_set_idx][
                                joint_idx
                            ][0] = float(smoothed_x[i, 0])  # Position only
                            frame_data["keypoints"][0]["keypoints"][keypoint_set_idx][
                                joint_idx
                            ][1] = float(smoothed_y[i, 0])  # Position only

            output_folder = os.path.join(
                output_base,
                subject,
                f"{action_group}_({'C' + str(camera + 1)})",
                "kalman",
            )
            save_filtered_keypoints(output_folder, json_path, pred_keypoints)

print("Kalman filtering complete.")

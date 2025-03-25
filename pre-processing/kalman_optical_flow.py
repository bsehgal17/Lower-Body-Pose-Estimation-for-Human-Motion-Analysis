import json
import os
import numpy as np
import cv2
from filterpy.kalman import KalmanFilter
from video_info import extract_video_info
import config
from joint_enum import PredJoints


def initialize_kalman_filter():
    """Initialize a Kalman Filter for tracking keypoints."""
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.F = np.array(
        [
            [1, 1, 0, 0],  # State transition
            [0, 1, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 0, 1],
        ]
    )
    kf.H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
    kf.P *= 1000
    kf.R = np.eye(2) * 5
    kf.Q = np.eye(4) * 0.1
    return kf


def compute_optical_flow(prev_frame, next_frame, prev_keypoints):
    """Uses Optical Flow to estimate motion correction."""
    prev_pts = np.array(prev_keypoints, dtype=np.float32).reshape(-1, 1, 2)
    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_frame, next_frame, prev_pts, None
    )

    corrected_keypoints = []
    for i, (new_pt, is_tracked) in enumerate(zip(next_pts, status)):
        if is_tracked:
            corrected_keypoints.append([float(new_pt[0][0]), float(new_pt[0][1])])
        else:
            corrected_keypoints.append(None)

    return corrected_keypoints


def correct_wrong_keypoints(video_frames, keypoints, alpha=0.3):
    """Fixes wrongly detected keypoints using Optical Flow, Kalman Filtering, and Interpolation."""
    num_frames = len(video_frames)
    corrected_keypoints = keypoints.copy()

    kf_x, kf_y = initialize_kalman_filter(), initialize_kalman_filter()

    for frame_idx in range(1, num_frames):
        prev_frame = cv2.cvtColor(video_frames[frame_idx - 1], cv2.COLOR_BGR2GRAY)
        next_frame = cv2.cvtColor(video_frames[frame_idx], cv2.COLOR_BGR2GRAY)

        for keypoint_group in corrected_keypoints[frame_idx]["keypoints"]:
            for keypoint_set in keypoint_group["keypoints"]:
                prev_keypoints = [
                    kp[:2] for kp in keypoint_set if kp is not None and kp[2] > 0.2
                ]

                if len(prev_keypoints) > 0:
                    corrected_points = compute_optical_flow(
                        prev_frame, next_frame, prev_keypoints
                    )

                    for joint_idx, corrected_kp in enumerate(corrected_points):
                        if keypoint_set[joint_idx][2] < 0.2:
                            if corrected_kp is not None:
                                keypoint_set[joint_idx] = [
                                    corrected_kp[0],
                                    corrected_kp[1],
                                    0.7,
                                ]

                        # Apply Kalman Filtering
                        kf_x.predict()
                        kf_y.predict()
                        kf_x.update([keypoint_set[joint_idx][0]])
                        kf_y.update([keypoint_set[joint_idx][1]])

                        # Blend Optical Flow & Kalman results
                        keypoint_set[joint_idx][0] = (
                            alpha * kf_x.x[0] + (1 - alpha) * corrected_kp[0]
                        )
                        keypoint_set[joint_idx][1] = (
                            alpha * kf_y.x[0] + (1 - alpha) * corrected_kp[1]
                        )

    return corrected_keypoints


def save_corrected_keypoints(original_json_path, corrected_keypoints):
    corrected_json_path = original_json_path.replace(".json", "_corrected.json")
    with open(corrected_json_path, "w") as f:
        json.dump(corrected_keypoints, f, indent=4)
    print(f"Corrected keypoints saved to: {corrected_json_path}")


base_path = config.VIDEO_FOLDER

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

            # Load keypoints
            with open(json_path, "r") as f:
                pred_keypoints = json.load(f)

            # Load video frames for Optical Flow
            video_path = json_path.replace(".json", ".mp4")
            cap = cv2.VideoCapture(video_path)
            frames = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            cap.release()

            corrected_keypoints = correct_wrong_keypoints(frames, pred_keypoints)
            save_corrected_keypoints(json_path, corrected_keypoints)

print("Processing complete.")

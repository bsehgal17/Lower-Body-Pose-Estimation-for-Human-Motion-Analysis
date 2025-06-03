"""
This script processes video files to detect and estimate human pose, and then saves the results in JSON format.

The script performs the following steps:
1. Initializes the necessary models for object detection, pose estimation, and visualization.
2. Retrieves all video files from the specified folder.
3. For each video, processes each frame to detect human poses.
4. Saves the detected keypoints for each frame in a JSON file.
5. Visualizes the detected poses on the frames and displays them.
"""

import cv2
import os
from utils.config import Config
from utils.utils import (
    get_video_files,
    frame_generator,
    save_keypoints_to_json,
    combine_keypoints,
)
from pose_estimation.detector import Detector
from pose_estimation.pose_estimator import PoseEstimator
from pose_estimation.visualization import PoseVisualizer


def detect_and_visualize_pose(config: Config):
    # Initialize models with config
    detector = Detector(config)
    pose_estimator = PoseEstimator(config)
    visualizer = PoseVisualizer(pose_estimator, config)

    # Get all videos from the folder
    video_files = get_video_files(config.paths.VIDEO_FOLDER)
    if not video_files:
        print("No video files found in the folder!")
        return

    for video_path in video_files:
        video_data = []
        frames_list = []
        rel_path = os.path.relpath(video_path, config.paths.VIDEO_FOLDER)
        video_dir_structure = os.path.dirname(rel_path)
        video_name_with_ex = os.path.basename(video_path)
        video_name = os.path.splitext(video_name_with_ex)[0]
        save_dir = os.path.join(
            config.paths.OUTPUT_DIR, video_dir_structure, video_name
        )
        os.makedirs(save_dir, exist_ok=True)
        output_video_file = os.path.join(save_dir, video_name_with_ex)
        print(f"\nProcessing Video: {video_name}")

        for frame_idx, frame in enumerate(frame_generator(video_path)):
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect humans
            bboxes = detector.detect_humans(frame_bgr)

            # Pose estimation
            data_samples, pose_results = pose_estimator.estimate_pose(frame, bboxes)

            combine_keypoints(pose_results, frame_idx, video_data, bboxes)
            visualizer.visualize_pose(frame, data_samples, frame_idx, frames_list)
            # visualizer.visualize_lower_points(frame, data_samples, frame_idx, frames_list, joints_to_visualize=[11, 12, 13, 14, 15, 16])

        # After processing all frames, save the entire video data
        save_keypoints_to_json(video_data, save_dir, video_name)
        visualizer.create_video_from_frames(frames_list, output_video_file)
        print("Processing complete. All results are saved in the output directory.")

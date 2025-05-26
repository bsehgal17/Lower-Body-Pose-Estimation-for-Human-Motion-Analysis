"""
This script processes video files to detect and estimate human pose, and then saves the results in JSON format.

The script performs the following steps:
1. Initializes the necessary models for object detection, pose estimation, and visualization.
2. Retrieves all video files from the specified folder.
3. For each video, processes each frame to detect human poses.
4. Saves the detected keypoints for each frame in a JSON file.
5. Visualizes the detected poses on the frames and displays them.

Dependencies:
- cv2: OpenCV for reading and processing video frames.
- config: Configuration file containing paths to the video folder and output directory.
- utils: Utility functions for handling video files, frames, and saving results.
- detector: Module for initializing the object detector and detecting humans.
- pose_estimator: Module for initializing the pose estimator and estimating poses.
- visualization: Module for visualizing pose estimation results."""

import cv2
import os
from config import VIDEO_FOLDER, OUTPUT_DIR
from utils import get_video_files, frame_generator, save_keypoints_to_json, combine_keypoints
from detector import init_object_detector, detect_humans
from pose_estimator import init_pose_estimator, estimate_pose
from visualization import init_visualizer, visualize_lower_points, create_video_from_frames, visualize_pose

# Initialize models
detector = init_object_detector()
pose_estimator = init_pose_estimator()
visualizer = init_visualizer(pose_estimator)

# Get all videos from the folder
video_files = get_video_files(VIDEO_FOLDER)
if not video_files:
    print("No video files found in the folder!")
    exit()


for video_path in video_files:
    video_data = []
    frames_list = []
    parent_folder = video_path.split("test_videos/")[-1].split("/")[0]
    subject_name = video_path.split("test_videos/")[-1].split("/")[1]
    video_name = os.path.basename(video_path)
    save_dir = os.path.join(
        OUTPUT_DIR, parent_folder, subject_name, os.path.splitext(video_name)[0])
    os.makedirs(save_dir, exist_ok=True)
    output_video_file = os.path.join(save_dir, video_name)
    print(f"\nProcessing Video: {video_name}")

    for frame_idx, frame in enumerate(frame_generator(video_path)):
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect humans
        bboxes = detect_humans(detector, frame_bgr)

        # Pose estimation
        data_samples, pose_results = estimate_pose(
            pose_estimator, frame, bboxes)

        visualizer = init_visualizer(pose_estimator)

        combine_keypoints(pose_results, frame_idx, video_data, bboxes)
        visualize_pose(frame, data_samples, visualizer,
                       frame_idx, frames_list)
        # visualize_lower_points(frame=frame, data_samples=data_samples, frame_idx=frame_idx,
        #                        frames_list=frames_list, joints_to_visualize=[11, 12, 13, 14, 15, 16])

    # After processing all frames, save the entire video data
    save_keypoints_to_json(video_data, save_dir, video_name)
    create_video_from_frames(frames_list, output_video_file)
    print("Processing complete. All results are saved in the output directory.")

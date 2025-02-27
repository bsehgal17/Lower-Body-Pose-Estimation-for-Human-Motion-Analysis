"""
This script evaluates the performance of keypoint detection models by comparing ground truth keypoints 
and predicted keypoints, specifically in the context of gait and pose analysis.

It performs the following steps:
1. Extracts ground truth keypoints from a provided MAT file.
2. Extracts predicted keypoints from a provided JSON file.
3. Computes various metrics to evaluate the accuracy of predicted keypoints compared to the ground truth.

Dependencies:
- extract_ground_truth: Module for extracting ground truth keypoints from the MAT file.
- extract_predictions: Module for extracting predicted keypoints from the JSON file.
- compute_metrics: Module for computing and displaying evaluation metrics.
- config: Configuration file containing paths to the MAT and JSON files."""

import os
from get_gt_keypoint import extract_ground_truth
from extract_predicted_points import extract_predictions
from compute_metrics import compute_metrics
from video_info import extract_video_info
import config
from visualize_gt_pred import plot_gt_pred

# Define base path
base_path = config.VIDEO_FOLDER

# Walk through all video files in the base directory
for root, dirs, files in os.walk(base_path):
    for file in files:
        video_info = extract_video_info(file, root)
        if video_info:

            subject, action, camera = video_info
            action_group = action.replace(
                ' ', '_')  # Replaces space with underscore
            json_path = os.path.join(r"C:\Users\BhavyaSehgal\Downloads\bhavya_1st_sem\humaneva\rtmw_results", subject,
                                     f"{action_group}_({'C' + str(camera + 1)})", f"{action_group}_({'C' + str(camera + 1)})/{action_group}_({'C' + str(camera + 1)})".replace(' ', '') + ".json")

            print(
                f"Processing: Subject={subject}, Action={action_group}, Camera={camera + 1}")

            # Extract ground truth keypoints
            gt_keypoints = extract_ground_truth(
                config.CSV_FILE, subject, action, camera)

            sync_frame = config.SYNC_DATA.get(
                subject, {}).get(action, None)[camera]
            frame_range = (sync_frame, (sync_frame+len(gt_keypoints)))
           
            # Extract predicted keypoints
            pred_keypoints = extract_predictions(json_path, frame_range)

            plot_gt_pred(gt_keypoints,pred_keypoints,root,file,[frame_range[0],frame_range[1]])

            # Compute and display metrics
            compute_metrics(gt_keypoints, pred_keypoints)

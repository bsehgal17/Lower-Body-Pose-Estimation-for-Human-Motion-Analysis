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

# Define base path
base_path = config.VIDEO_FOLDER

# Walk through all video files in the base directory
for root, dirs, files in os.walk(base_path):
    for file in files:
        video_info = extract_video_info(file, root)
        if video_info:
            subject, action_group, camera = video_info

            print(
                f"Processing: Subject={subject}, Action={action_group}, Camera={camera + 1}")

            # Extract ground truth keypoints
            gt_keypoints = extract_ground_truth(
                config.CSV_FILE, subject, action_group, camera)

            # Extract predicted keypoints
            pred_keypoints = extract_predictions(
                config.JSON_FILE, subject, action_group, camera)

            # Compute and display metrics
            compute_metrics(gt_keypoints, pred_keypoints)

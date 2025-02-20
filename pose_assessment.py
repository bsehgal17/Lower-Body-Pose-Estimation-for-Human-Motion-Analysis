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

from extract_ground_truth import extract_ground_truth
from extract_predictions import extract_predictions
from compute_metrics import compute_metrics
import config

# Extract ground truth keypoints
gt_keypoints = extract_ground_truth(config.MAT_FILE)

# Extract predicted keypoints
pred_keypoints = extract_predictions(config.JSON_FILE)

# Compute and display metrics
compute_metrics(gt_keypoints, pred_keypoints)

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
from compute_pck import compute_pck
from video_info import extract_video_info
import config
import pandas as pd
from visualize_gt_pred import plot_gt_pred
from visualize_pred_points import visualize_predictions

# Define base path
base_path = config.VIDEO_FOLDER

results = []
lower_body_joints = [
    "LEFT_HIP",
    "RIGHT_HIP",
    "LEFT_KNEE",
    "RIGHT_KNEE",
    "LEFT_ANKLE",
    "RIGHT_ANKLE",
]
# Walk through all video files in the base directory
for root, dirs, files in os.walk(base_path):
    for file in files:
        video_info = extract_video_info(file, root)
        if video_info:
            subject, action, camera = video_info
            action_group = action.replace(" ", "_")  # Replaces space with underscore
            json_path = os.path.join(
                r"C:\Users\BhavyaSehgal\Downloads\bhavya_1st_sem\humaneva\rtw_l",
                subject,
                f"{action_group}_({'C' + str(camera + 1)})",
                f"{action_group}_({'C' + str(camera + 1)})/{action_group}_({'C' + str(camera + 1)})".replace(
                    " ", ""
                )
                + ".json",
            )

            print(
                f"Processing: Subject={subject}, Action={action_group}, Camera={camera + 1}"
            )

            # Extract ground truth keypoints
            gt_keypoints = extract_ground_truth(
                config.CSV_FILE,
                subject,
                action,
                camera,
            )

            sync_frame = config.SYNC_DATA.get(subject, {}).get(action, None)[camera]
            frame_range = (sync_frame, (sync_frame + len(gt_keypoints)))

            # Extract predicted keypoints
            pred_keypoints = extract_predictions(
                json_path,
                frame_range,
            )

            # Ensure both gt_keypoints and pred_keypoints have the same length
            min_length = min(len(gt_keypoints), len(pred_keypoints))

            # Trim both arrays to the minimum length
            gt_keypoints = gt_keypoints[:min_length]
            pred_keypoints = pred_keypoints[:min_length]

            # Plot ground truth and predicted keypoints
            # plot_gt_pred(
            #     csv_path=config.CSV_FILE,
            #     subject=subject,
            #     action=action,
            #     camera=camera,
            #     root=root,
            #     video_name=file,
            #     frame_ranges=[frame_range[0], frame_range[1]],
            # )
            # video_path = os.path.join(root, file)
            # visualize_predictions(
            #     video_path=video_path, json_file=json_path, frame_range=frame_range
            # )
            pck2 = compute_pck(
                gt_keypoints,
                pred_keypoints,
                threshold=0.02,
                # joints_to_evaluate=lower_body_joints,
            )
            pck1 = compute_pck(
                gt_keypoints,
                pred_keypoints,
                threshold=0.01,
                # joints_to_evaluate=lower_body_joints,
            )
            pck5 = compute_pck(
                gt_keypoints,
                pred_keypoints,
                threshold=0.05,
                # joints_to_evaluate=lower_body_joints,
            )

            print("--- Results ---")
            print(f"PCK@0.2: {pck2:.2f}%")
            print(f"PCK@0.1: {pck1:.2f}%")
            print(f"PCK@0.05: {pck5:.2f}%")

            # Append results
            results.append([subject, action_group, camera + 1, pck1, pck2, pck5])

# Convert results to a DataFrame
df = pd.DataFrame(
    results, columns=["Subject", "Action", "Camera", "PCK@0.01", "PCK@0.02", "PCK@0.05"]
)

# Save to Excel
excel_path = r"C:\Users\BhavyaSehgal\Downloads\bhavya_1st_sem\humaneva\rtmw_results\comparsion_excels/rtmw_l_whole.xlsx"
df.to_excel(excel_path, index=False)
print(f"Metrics saved to {excel_path}")

# analysis/assessment_runner.py
import os
import pandas as pd
import logging
from typing import List, Dict, Any

from config.base import GlobalConfig
from Compute_metrics.get_gt_keypoint import extract_ground_truth
from Compute_metrics.extract_predicted_points import extract_predictions
from Compute_metrics.compute_pck import compute_pck
from utils.video_info import extract_video_info
from noise.rescale_pred import (
    get_video_resolution,
    rescale_keypoints,
)  # Ensure correct path

logger = logging.getLogger(__name__)


def run_pose_assessment_pipeline(config: GlobalConfig):
    """
    Runs the pose assessment pipeline, comparing predicted poses with ground truth.

    Args:
        config (GlobalConfig): The global configuration object.
    """
    logger.info("Starting pose assessment pipeline...")

    original_video_base = config.paths.video_folder  # Assuming original videos are here
    degraded_data_base = (
        config.paths.output_dir
    )  # Assuming processed/degraded data (JSONs) are here
    csv_file = config.paths.csv_file
    sync_data = config.sync_data.data  # Access the dictionary directly

    results = []
    # Define lower body joints (can be made configurable in config.analysis)
    lower_body_joints = [
        "LEFT_HIP",
        "RIGHT_HIP",
        "LEFT_KNEE",
        "RIGHT_KNEE",
        "LEFT_ANKLE",
        "RIGHT_ANKLE",
    ]  # Assuming PredJoints names or similar mapping

    # Iterate through the processed/degraded video results (JSON files)
    # This logic assumes the JSON files are structured in a predictable way
    for root, _, files in os.walk(degraded_data_base):
        for file in files:
            # Look for JSON files that are *not* filtered ones (if you want to assess raw predictions)
            # Adjust this filter depending on whether you want to assess raw, noisy, or filtered results
            if file.endswith(".json") and not "_filtered" in file:
                json_path = os.path.join(root, file)

                # Derive video info from the JSON path
                # This part is critical and depends on your specific output structure
                # Example: `output_dir/subject/action/camera/video_name.json`

                # You'll need a way to parse `root` to get subject, action, camera
                # For example, if root is like 'degraded_data_base/S1/Walking_1_(C1)':

                # Placeholder for extracting info from path
                # In a real scenario, you might have a dedicated utility like:
                # video_info = parse_info_from_output_path(root, file)

                # For now, let's try to infer from the file name if extract_video_info works
                # Assumes file name is like 'Walking_1_(C1).json' and root contains subject

                # Try to get video_info from the original video's file name
                # This needs to map the JSON path back to the original video file name
                # You might need to store original video names in your JSONs or have a consistent mapping.

                # A more robust way: store the original video path/name in the JSON output from `detect_and_visualize_pose`
                # For now, assuming `extract_video_info` can work with JSON filenames and paths
                video_info = extract_video_info(file, root)
                if not video_info:
                    logger.warning(
                        f"Could not extract video info from {json_path}. Skipping assessment."
                    )
                    continue
                subject, action, camera_idx = (
                    video_info  # Adapt based on extract_video_info output
                )
                action_group = action.replace(" ", "_")
                cam_name = f"C{camera_idx + 1}"  # HumanEva cameras are 0-indexed in paths but C1, C2, C3 in names

                logger.info(
                    f"Assessing: Subject={subject}, Action={action_group}, Camera={cam_name} from {json_path}"
                )

                try:
                    # Ground truth paths
                    # This relies on a strict path structure of original videos
                    original_video_path = os.path.join(
                        original_video_base,
                        subject,
                        "Image_Data",
                        f"{action}_{cam_name}.avi",  # Adjust filename format if needed
                    )

                    # Extract ground truth keypoints
                    gt_keypoints = extract_ground_truth(
                        csv_file, subject, action, camera_idx
                    )

                    # Get sync frame from config
                    sync_frame_tuple = sync_data.get(subject, {}).get(action, None)
                    if not sync_frame_tuple:
                        logger.warning(
                            f"Sync data not found for {subject}, {action}. Skipping {json_path}."
                        )
                        continue
                    sync_frame = sync_frame_tuple[
                        camera_idx
                    ]  # Adjust index based on camera_idx in your data

                    frame_range = (sync_frame, sync_frame + len(gt_keypoints))

                    # Extract predicted keypoints
                    pred_keypoints_org = extract_predictions(json_path, frame_range)

                    # Rescale predicted keypoints to match original resolution
                    orig_w, orig_h = get_video_resolution(original_video_path)

                    # The degraded video resolution needs to be from the actual video used for pose estimation
                    # If you applied noise, you might have changed resolution. This assumes it's available.
                    # Or, the JSON should store the resolution it was processed at.
                    # For now, let's assume get_video_resolution can handle the noisy video path as well.
                    # This assumes the JSON is in a folder containing the corresponding noisy video.

                    # A better way: the JSON should store metadata including the resolution it was processed at.
                    # For now, derive path to degraded video:
                    degraded_video_path = os.path.join(
                        os.path.dirname(json_path), f"{os.path.splitext(file)[0]}.avi"
                    )  # Assuming .avi
                    degraded_w, degraded_h = get_video_resolution(degraded_video_path)

                    scale_x = orig_w / degraded_w
                    scale_y = orig_h / degraded_h

                    pred_keypoints = rescale_keypoints(
                        pred_keypoints_org, scale_x, scale_y
                    )

                    # Align keypoints
                    min_len = min(len(gt_keypoints), len(pred_keypoints))
                    gt_keypoints_aligned = gt_keypoints[:min_len]
                    pred_keypoints_aligned = pred_keypoints[:min_len]

                    # Compute PCK metrics (thresholds can be configured)
                    pck_thresholds = config.analysis.get(
                        "pck_thresholds", [0.005, 0.01, 0.02, 0.05]
                    )  # Add to config

                    pck_results = {}
                    for threshold in pck_thresholds:
                        pck_results[f"PCK@{threshold:.3f}"] = compute_pck(
                            gt_keypoints_aligned,
                            pred_keypoints_aligned,
                            threshold=threshold,
                        )
                        logger.info(
                            f"  PCK@{threshold:.3f}: {pck_results[f'PCK@{threshold:.3f}']:.2f}%"
                        )

                    results.append(
                        [subject, action_group, camera_idx + 1]
                        + list(pck_results.values())
                    )

                except FileNotFoundError as fnfe:
                    logger.error(
                        f"Required file not found for assessment: {fnfe}. Skipping {json_path}"
                    )
                except Exception as e:
                    logger.error(f"Error during assessment of {json_path}: {e}")
                    # Optionally, log the traceback for more detail: logger.exception(...)

    if results:
        # Save all results to Excel
        columns = ["Subject", "Action", "Camera"] + [
            f"PCK@{t:.3f}" for t in pck_thresholds
        ]
        df = pd.DataFrame(results, columns=columns)

        excel_output_path = (
            config.paths.output_dir
        )  # Assuming output_dir is where you want to save the assessment excel
        excel_filename = config.analysis.get(
            "excel_output_filename", "combined_assessment_results.xlsx"
        )  # Add to config
        full_excel_path = os.path.join(excel_output_path, excel_filename)

        os.makedirs(
            os.path.dirname(full_excel_path), exist_ok=True
        )  # Ensure path exists
        df.to_excel(full_excel_path, index=False)
        logger.info(f"Assessment metrics saved to {full_excel_path}")
    else:
        logger.info("No assessment results were generated.")

    logger.info("Pose assessment pipeline finished.")

import os
import logging
import pandas as pd
import re
from config.base import GlobalConfig
from evaluation.overall_pck import OverallPCKCalculator
from evaluation.pck_evaluator import PCKEvaluator
from evaluation.get_gt_keypoint import GroundTruthLoader
from evaluation.extract_predicted_points import PredictionExtractor
from utils.video_info import extract_video_info  # Optional, fallback to manual parsing
from utils.rescale_pred import get_video_resolution, rescale_keypoints

logger = logging.getLogger(__name__)


def assess_single_sample(
    subject, action, camera_idx, json_path, config, csv_file_path, original_video_base
):
    """Run assessment for a single video sample."""
    try:
        cam_name = f"C{camera_idx + 1}"
        original_video_path = os.path.join(
            original_video_base, subject, "Image_Data", f"{action}_{cam_name}.avi"
        )

        gt_loader = GroundTruthLoader(csv_file_path)
        gt_keypoints = gt_loader.get_keypoints(
            subject, action, camera_idx, chunk="chunk0"
        )

        sync_frame_tuple = config.sync_data.data.get(subject, {}).get(action, None)
        if not sync_frame_tuple:
            logger.warning(f"Missing sync data for {subject}, {action}.")
            return None
        sync_frame = sync_frame_tuple[camera_idx]
        frame_range = (sync_frame, sync_frame + len(gt_keypoints))

        pred_loader = PredictionExtractor(json_path, file_format="json")
        pred_keypoints_org = pred_loader.get_keypoint_array(frame_range=frame_range)

        testing_video_path = os.path.join(
            os.path.dirname(json_path),
            f"{os.path.splitext(os.path.basename(json_path))[0]}.avi",
        )

        orig_w, orig_h = get_video_resolution(original_video_path)

        if os.path.exists(testing_video_path):
            testing_w, testing_h = get_video_resolution(testing_video_path)
            if (testing_w, testing_h) != (orig_w, orig_h):
                scale_x, scale_y = orig_w / testing_w, orig_h / testing_h
                pred_keypoints = rescale_keypoints(pred_keypoints_org, scale_x, scale_y)
            else:
                pred_keypoints = pred_keypoints_org
        else:
            logger.warning(
                f"Degraded video not found at {testing_video_path}, using raw predictions."
            )
            pred_keypoints = pred_keypoints_org

        min_len = min(len(gt_keypoints), len(pred_keypoints))
        gt_keypoints_aligned = gt_keypoints[:min_len]
        pred_keypoints_aligned = pred_keypoints[:min_len]

        return gt_keypoints_aligned, pred_keypoints_aligned

    except FileNotFoundError as fnfe:
        logger.error(f"File not found: {fnfe}")
    except Exception as e:
        logger.error(f"Error in assess_single_sample: {e}")

    return None


def run_pose_assessment_pipeline(config: GlobalConfig):
    logger.info("Starting pose assessment pipeline...")

    original_video_base = config.paths.video_folder
    tested_data_base = config.paths.output_dir
    csv_file_path = config.paths.csv_file
    pck_thresholds = config.analysis.get("pck_thresholds", [0.005, 0.01, 0.02, 0.05])
    excel_filename = config.analysis.get(
        "excel_output_filename", "combined_assessment_results.xlsx"
    )
    excel_output_path = os.path.join(config.paths.output_dir, excel_filename)

    evaluator = PCKEvaluator(excel_output_path)

    for root, _, files in os.walk(tested_data_base):
        for file in files:
            if file.endswith(".json"):
                json_path = os.path.join(root, file)

                # Generalized extraction from folder path
                path_parts = os.path.normpath(root).split(os.sep)
                subject, action, camera_idx = None, None, None

                if len(path_parts) >= 2:
                    subject = path_parts[-2]
                    action_cam = path_parts[-1]
                    match = re.match(r"(.+)_\(C(\d+)\)", action_cam)
                    if match:
                        action = match.group(1).replace("_", " ")
                        camera_idx = int(match.group(2)) - 1

                if not subject or not action or camera_idx is None:
                    logger.warning(
                        f"Could not infer subject/action/camera from: {root}. Skipping."
                    )
                    continue

                action_group = action.replace(" ", "_")
                logger.info(
                    f"Assessing: Subject={subject}, Action={action_group}, Camera=C{camera_idx + 1}"
                )

                result = assess_single_sample(
                    subject,
                    action,
                    camera_idx,
                    json_path,
                    config,
                    csv_file_path,
                    original_video_base,
                )

                if result is None:
                    continue

                gt_keypoints, pred_keypoints = result

                for threshold in pck_thresholds:
                    calculator = OverallPCKCalculator(threshold=threshold)
                    evaluator.evaluate_overall(
                        calculator,
                        gt_keypoints,
                        pred_keypoints,
                        subject,
                        action_group,
                        camera_idx + 1,
                    )

    evaluator.save()
    logger.info("Pose assessment pipeline finished.")

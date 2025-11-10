import os
import logging
import pickle
import numpy as np
from config.pipeline_config import PipelineConfig
from config.global_config import GlobalConfig
from dataset_files.HumanSC3D.humansc3d_metadata import (
    get_humansc3d_metadata_from_video,
    get_humansc3d_gt_path,
)
from dataset_files.HumanSC3D.humansc3d_gt_loader import HumanSC3DGroundTruthLoader
from evaluation.get_pred_keypoint import PredictionLoader
from utils.import_utils import import_class_from_string
from evaluation.generic_evaluator import MetricsEvaluator, run_assessment

logger = logging.getLogger(__name__)


def humansc3d_data_loader(
    pred_pkl_path,
    pipeline_config: PipelineConfig,
    global_config: GlobalConfig,
    min_bbox_confidence,
    min_keypoint_confidence,
):
    """
    Loads and prepares GT and prediction data for a single HumanSC3D sample.
    Returns (gt_keypoints, gt_bboxes, gt_scores, pred_keypoints, pred_bboxes, pred_scores, sample_info).
    The GT data will have placeholder bboxes and scores since the dataset does not provide them.

    Args:
        pred_pkl_path (str): Path to prediction pickle file
        pipeline_config (PipelineConfig): Pipeline configuration
        global_config (GlobalConfig): Global configuration
        min_bbox_confidence (float): Minimum bounding box confidence threshold for filtering
        min_keypoint_confidence (float): Minimum keypoint confidence threshold for filtering
    """
    try:
        json_path = pred_pkl_path.replace(".pkl", ".json")
        metadata = get_humansc3d_metadata_from_video(json_path)
        if not metadata:
            logger.warning(
                f"Could not parse metadata from {os.path.basename(json_path)}"
            )
            return None

        subject, action, camera = (
            metadata["subject"],
            metadata["action"],
            metadata["camera"],
        )

        # --- GT Loading Logic (HumanSC3D specific) ---
        original_video_base = os.path.join(
            global_config.paths.input_dir, pipeline_config.paths.dataset
        )

        # Construct the video path to derive GT path
        original_video_path = os.path.join(
            original_video_base, subject, "videos", camera, f"{action}.mp4"
        )

        # Get ground truth path using the metadata helper
        try:
            gt_json_path = get_humansc3d_gt_path(original_video_path)
        except ValueError as e:
            logger.warning(
                f"Could not determine GT path for {original_video_path}: {e}"
            )
            return None

        if not os.path.exists(gt_json_path):
            logger.warning(f"Ground truth file not found: {gt_json_path}")
            return None

        # Cache GT data as pickle for faster loading
        gt_pkl_name = f"{subject}_{action}_{camera}_gt.pkl"
        gt_pkl_folder = os.path.join(os.path.dirname(gt_json_path), "pickle_files")
        os.makedirs(gt_pkl_folder, exist_ok=True)
        gt_pkl_path = os.path.join(gt_pkl_folder, gt_pkl_name)

        if not os.path.exists(gt_pkl_path):
            loader = HumanSC3DGroundTruthLoader(gt_json_path)
            keypoints = loader.get_keypoints_from_path(gt_json_path)
            if keypoints is not None:
                with open(gt_pkl_path, "wb") as f:
                    pickle.dump({"keypoints": keypoints}, f)
            else:
                logger.warning(f"Could not load GT keypoints from {gt_json_path}")
                return None

        with open(gt_pkl_path, "rb") as f:
            gt_data = pickle.load(f)
        gt_keypoints = gt_data["keypoints"]

        # If GT keypoints are None or empty, skip this sample
        if gt_keypoints is None or (
            isinstance(gt_keypoints, (list, np.ndarray)) and len(gt_keypoints) == 0
        ):
            logger.warning(
                f"Skipping {pred_pkl_path} due to missing or empty GT keypoints."
            )
            return None

        # --- Prediction Loading and Preprocessing ---
        pred_loader = PredictionLoader(
            pred_pkl_path,
            pipeline_config,
        )
        pred_keypoints, pred_bboxes, pred_scores = pred_loader.get_filtered_predictions(
            subject, action, int(camera), original_video_path
        )

        # If pred_keypoints is None or empty, also skip (optional, for robustness)
        if pred_keypoints is None or (
            isinstance(pred_keypoints, (list, np.ndarray)) and len(pred_keypoints) == 0
        ):
            logger.warning(
                f"Skipping {pred_pkl_path} due to missing or empty predicted keypoints."
            )
            return None

        min_len = min(len(gt_keypoints), len(pred_keypoints))

        # Create placeholder lists for GT bboxes and scores for MAPCalculator
        gt_bboxes = [None] * min_len
        gt_scores = [1.0] * min_len

        sample_info = {
            "subject": subject,
            "action": action,
            "camera": camera,
        }

        # Return all the separate lists in the expected order
        return (
            gt_keypoints[:min_len],
            gt_bboxes,
            gt_scores,
            pred_keypoints[:min_len],
            pred_bboxes[:min_len],
            pred_scores[:min_len],
            sample_info,
        )

    except Exception as e:
        logger.error(f"Assessment error for {pred_pkl_path}: {e}")
        return None


def run_humansc3d_assessment(
    pipeline_config: PipelineConfig,
    global_config: GlobalConfig,
    output_dir: str,
    input_dir: str,
    min_bbox_confidence=None,
    min_keypoint_confidence=None,
):
    """
    Run HumanSC3D assessment with configurable confidence filtering.

    Args:
        pipeline_config (PipelineConfig): Pipeline configuration
        global_config (GlobalConfig): Global configuration
        output_dir (str): Output directory for results
        input_dir (str): Input directory for data
        min_bbox_confidence (float or None): Minimum bounding box confidence threshold for filtering
        min_keypoint_confidence (float or None): Minimum keypoint confidence threshold for filtering
    """
    gt_enum_class = import_class_from_string(pipeline_config.dataset.joint_enum_module)
    pred_enum_class = import_class_from_string(pipeline_config.dataset.keypoint_format)

    pred_root = (
        pipeline_config.evaluation.input_dir or pipeline_config.detect.output_dir
    )
    # Pass the output path to the evaluator
    evaluator = MetricsEvaluator(output_path=output_dir)

    # Define the grouping keys for HumanSC3D
    grouping_keys = ["subject", "action", "camera"]

    # Create a wrapper function that includes only the confidence parameters
    def data_loader_with_options(pred_pkl_path, pipeline_config, global_config):
        return humansc3d_data_loader(
            pred_pkl_path,
            pipeline_config,
            global_config,
            min_bbox_confidence=min_bbox_confidence,
            min_keypoint_confidence=min_keypoint_confidence,
        )

    run_assessment(
        evaluator=evaluator,
        pipeline_config=pipeline_config,
        global_config=global_config,
        input_dir=pred_root,
        output_dir=output_dir,
        gt_enum_class=gt_enum_class,
        pred_enum_class=pred_enum_class,
        data_loader_func=data_loader_with_options,
        grouping_keys=grouping_keys,  # Pass the grouping keys
    )

    logger.info("HumanSC3D assessment completed.")

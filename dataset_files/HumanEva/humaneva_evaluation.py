import os
import logging
import pickle
from config.pipeline_config import PipelineConfig
from config.global_config import GlobalConfig
from dataset_files.HumanEva.humaneva_metadata import get_humaneva_metadata_from_video
from dataset_files.HumanEva.get_gt_keypoint import GroundTruthLoader
from dataset_files.HumanEva.get_pred_keypoint import PredictionLoader
from utils.import_utils import import_class_from_string
from evaluation.generic_evaluator import MetricsEvaluator, run_assessment

logger = logging.getLogger(__name__)


def _get_sync_offset(
    pipeline_config, subject: str, action: str, camera_idx: int
) -> int:
    """Get synchronization offset for proper frame alignment."""
    try:
        if hasattr(pipeline_config, "dataset") and hasattr(
            pipeline_config.dataset, "sync_data"
        ):
            sync_start = pipeline_config.dataset.sync_data["data"][subject][action][
                camera_idx
            ]
            return sync_start
    except (KeyError, AttributeError) as e:
        logger.warning(
            f"No sync data found for {subject}/{action}/camera_{camera_idx}: {e}"
        )
    return 0


def humaneva_data_loader(
    pred_pkl_path,
    pipeline_config: PipelineConfig,
    global_config: GlobalConfig,
    min_bbox_confidence,
    min_keypoint_confidence,
):
    """
    Loads and prepares GT and prediction data for a single HumanEva sample.
    Returns (gt_keypoints, gt_bboxes, gt_scores, pred_keypoints, pred_bboxes, pred_scores, sample_info).
    The GT data will have placeholder bboxes and scores since the dataset does not provide them.

    Args:
        pred_pkl_path (str): Path to prediction pickle file
        pipeline_config (PipelineConfig): Pipeline configuration
        global_config (GlobalConfig): Global configuration
        min_bbox_confidence (float): Minimum bounding box confidence threshold for filtering
        min_keypoint_confidence (float): Minimum keypoint confidence threshold for filtering
        enable_visualization (bool): Whether to generate visualization frames
        visualization_output_dir (str): Output directory for visualization frames
        frame_step (int): Step size for frame processing (1=every frame, 10=every 10th frame)
        visualize_gt (bool): Whether to include ground truth in visualizations
    """
    try:
        json_path = pred_pkl_path.replace(".pkl", ".json")
        metadata = get_humaneva_metadata_from_video(json_path)
        if not metadata:
            logger.warning(
                f"Could not parse metadata from {os.path.basename(json_path)}"
            )
            return None

        subject, action, camera_str = (
            metadata["subject"],
            metadata["action"],
            metadata["camera"],
        )
        camera_idx = int(camera_str[1:])

        safe_action_name = action.replace(" ", "_")

        # --- GT Loading Logic (HumanEva specific) ---
        original_video_base = os.path.join(
            global_config.paths.input_dir, pipeline_config.paths.dataset
        )
        csv_file_path = pipeline_config.paths.ground_truth_file

        gt_dir = os.path.dirname(csv_file_path)
        gt_pkl_name = f"{subject}_{safe_action_name}_{camera_str}_gt.pkl"
        gt_pkl_folder = os.path.join(gt_dir, "pickle_files")
        os.makedirs(gt_pkl_folder, exist_ok=True)
        gt_pkl_path = os.path.join(gt_pkl_folder, gt_pkl_name)

        if not os.path.exists(gt_pkl_path):
            loader = GroundTruthLoader(csv_file_path)
            keypoints = loader.get_keypoints(
                subject, action, camera_idx, chunk="chunk0"
            )
            with open(gt_pkl_path, "wb") as f:
                pickle.dump({"keypoints": keypoints}, f)

        with open(gt_pkl_path, "rb") as f:
            gt_data = pickle.load(f)
        gt_keypoints = gt_data["keypoints"]

        # --- Prediction Loading and Preprocessing ---
        original_video_path = os.path.join(
            original_video_base,
            subject,
            "Image_Data",
            f"{safe_action_name}_({camera_str}).avi",
        )

        # Remove confidence filtering: pass zero thresholds and weights
        pred_loader = PredictionLoader(
            pred_pkl_path,
            pipeline_config,
            min_bbox_confidence=0.0,
            min_keypoint_confidence=0.0,
            bbox_weight=0.0,
            keypoint_weight=0.0,
        )
        pred_keypoints, pred_bboxes, pred_scores = pred_loader.get_filtered_predictions(
            subject, action, camera_idx - 1, original_video_path
        )
        min_len = min(len(gt_keypoints), len(pred_keypoints))

        # Create placeholder lists for GT bboxes and scores for MAPCalculator
        gt_bboxes = [None] * min_len
        gt_scores = [1.0] * min_len

        sample_info = {
            "subject": subject,
            "action": safe_action_name,
            "camera": camera_idx,
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


def run_humaneva_assessment(
    pipeline_config: PipelineConfig,
    global_config: GlobalConfig,
    output_dir: str,
    input_dir: str,
    min_bbox_confidence=None,
    min_keypoint_confidence=None,
):
    """
    Run HumanEva assessment with configurable confidence filtering and optional visualization.

    Args:
        pipeline_config (PipelineConfig): Pipeline configuration
        global_config (GlobalConfig): Global configuration
        output_dir (str): Output directory for results
        input_dir (str): Input directory for data
        min_bbox_confidence (float or None): Minimum bounding box confidence threshold for filtering
        min_keypoint_confidence (float or None): Minimum keypoint confidence threshold for filtering
        enable_visualization (bool): Whether to generate visualization frames
        visualization_output_dir (str): Output directory for visualization frames
        frame_step (int): Step size for frame processing
        visualize_gt (bool): Whether to include ground truth in visualizations
    """
    gt_enum_class = import_class_from_string(pipeline_config.dataset.joint_enum_module)
    pred_enum_class = import_class_from_string(pipeline_config.dataset.keypoint_format)

    # Use config values if CLI arguments are None
    bbox_threshold = (
        min_bbox_confidence
        if min_bbox_confidence is not None
        else pipeline_config.confidence_filtering.min_bbox_confidence
    )
    keypoint_threshold = (
        min_keypoint_confidence
        if min_keypoint_confidence is not None
        else pipeline_config.confidence_filtering.min_keypoint_confidence
    )

    logger.info(
        f"Running HumanEva assessment with confidence filtering "
        f"(bbox >= {bbox_threshold}, keypoint >= {keypoint_threshold})"
    )

    # Visualization logic removed

    pred_root = (
        pipeline_config.evaluation.input_dir or pipeline_config.detect.output_dir
    )
    # Pass the output path to the evaluator
    evaluator = MetricsEvaluator(output_path=output_dir)

    # Define the grouping keys for HumanEva
    grouping_keys = ["subject", "action", "camera"]

    # Create a wrapper function that includes only the confidence parameters
    def data_loader_with_options(pred_pkl_path, pipeline_config, global_config):
        return humaneva_data_loader(
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

    logger.info("HumanEva assessment completed.")

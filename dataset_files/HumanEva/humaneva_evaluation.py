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


def _draw_keypoints_on_frame(
    frame, keypoints, color, draw_skeleton=True, point_radius=4
):
    """Draw keypoints and skeleton on frame."""
    import cv2
    import numpy as np

    if keypoints is None or len(keypoints) == 0:
        return frame

    annotated_frame = frame.copy()

    # Lower body skeleton connections (using standard pose estimation indices)
    skeleton_connections = [
        (11, 12),  # Left hip to right hip
        (11, 13),  # Left hip to left knee
        (13, 15),  # Left knee to left ankle
        (12, 14),  # Right hip to right knee
        (14, 16),  # Right knee to right ankle
    ]

    # Draw keypoints
    for i, keypoint in enumerate(keypoints):
        if (
            len(keypoint) >= 2
            and not np.isnan(keypoint[0])
            and not np.isnan(keypoint[1])
        ):
            x, y = int(keypoint[0]), int(keypoint[1])
            cv2.circle(annotated_frame, (x, y), point_radius, color, -1)
            # Add joint index as text
            cv2.putText(
                annotated_frame,
                str(i),
                (x + 5, y + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                color,
                1,
            )

    # Draw skeleton connections
    if draw_skeleton:
        for start_idx, end_idx in skeleton_connections:
            if (
                start_idx < len(keypoints)
                and end_idx < len(keypoints)
                and not np.isnan(keypoints[start_idx][0])
                and not np.isnan(keypoints[start_idx][1])
                and not np.isnan(keypoints[end_idx][0])
                and not np.isnan(keypoints[end_idx][1])
            ):
                start_point = (
                    int(keypoints[start_idx][0]),
                    int(keypoints[start_idx][1]),
                )
                end_point = (int(keypoints[end_idx][0]),
                             int(keypoints[end_idx][1]))
                cv2.line(annotated_frame, start_point, end_point, color, 2)

    return annotated_frame


def _generate_filtered_frames(
    pred_keypoints,
    gt_keypoints,
    original_video_path,
    output_dir,
    subject,
    action,
    camera_str,
    frame_step,
    visualize_gt,
    sync_offset,
):
    """Generate filtered frames with keypoint overlays."""
    import cv2
    import numpy as np

    # Create output directory
    video_output_dir = os.path.join(
        output_dir, f"{subject}_{action}_{camera_str}_filtered"
    )
    os.makedirs(video_output_dir, exist_ok=True)

    # Color scheme
    colors = {
        "gt": (0, 255, 0),  # Green for ground truth
        "pred": (255, 0, 0),  # Blue for predictions
    }

    # Open video
    cap = cv2.VideoCapture(original_video_path)
    if not cap.isOpened():
        logger.error(f"Could not open video: {original_video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Determine frame range
    start_frame = sync_offset
    end_frame = min(total_frames, len(pred_keypoints) + sync_offset)

    logger.info(
        f"Processing frames {start_frame} to {end_frame} (step={frame_step})")

    stats = {
        "total_frames_processed": 0,
        "frames_with_valid_predictions": 0,
        "frames_with_gt": 0,
    }

    # Process frames
    for frame_idx in range(start_frame, end_frame, frame_step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        # Calculate prediction index (accounting for sync offset)
        pred_idx = frame_idx - sync_offset

        annotated_frame = frame.copy()
        has_valid_predictions = False

        # Draw predictions
        if 0 <= pred_idx < len(pred_keypoints):
            pred_kpts = pred_keypoints[pred_idx]
            if pred_kpts is not None:
                annotated_frame = _draw_keypoints_on_frame(
                    annotated_frame, pred_kpts, colors["pred"]
                )
                has_valid_predictions = True

        # Draw ground truth
        if (
            visualize_gt
            and gt_keypoints is not None
            and 0 <= pred_idx < len(gt_keypoints)
        ):
            gt_kpts = gt_keypoints[pred_idx]
            if gt_kpts is not None and len(gt_kpts) > 0:
                # Convert GT keypoints to proper format if needed
                if isinstance(gt_kpts, list):
                    gt_kpts = np.array(gt_kpts)
                if gt_kpts.ndim == 1:
                    gt_kpts = gt_kpts.reshape(-1, 2)

                annotated_frame = _draw_keypoints_on_frame(
                    annotated_frame, gt_kpts, colors["gt"]
                )
                stats["frames_with_gt"] += 1

        # Add frame information
        info_text = [
            f"Frame: {frame_idx}",
            f"Pred Index: {pred_idx}",
            f"Subject: {subject}, Action: {action}, Camera: {camera_str}",
        ]

        y_offset = 30
        for text in info_text:
            cv2.putText(
                annotated_frame,
                text,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            y_offset += 25

        # Save frame
        frame_filename = f"frame_{frame_idx:06d}.jpg"
        frame_path = os.path.join(video_output_dir, frame_filename)
        cv2.imwrite(frame_path, annotated_frame)

        if has_valid_predictions:
            stats["frames_with_valid_predictions"] += 1

        stats["total_frames_processed"] += 1

    cap.release()

    # Save processing summary
    summary_path = os.path.join(video_output_dir, "processing_summary.json")
    summary_data = {
        "video_info": {
            "subject": subject,
            "action": action,
            "camera": camera_str,
            "total_frames": total_frames,
            "fps": fps,
            "sync_offset": sync_offset,
        },
        "statistics": stats,
    }

    import json

    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2)

    logger.info(
        f"Saved {stats['total_frames_processed']} frames to: {video_output_dir}"
    )


def humaneva_data_loader(
    pred_pkl_path,
    pipeline_config: PipelineConfig,
    global_config: GlobalConfig,
    min_bbox_confidence,
    min_keypoint_confidence,
    enable_visualization=False,
    visualization_output_dir=None,
    frame_step=1,
    visualize_gt=True,
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

        # Use PredictionLoader with confidence filtering parameters from config
        confidence_config = pipeline_config.confidence_filtering

        # Use config values if available and enabled, otherwise use function arguments
        if confidence_config and confidence_config.enabled:
            final_bbox_conf = confidence_config.min_bbox_confidence
            final_keypoint_conf = confidence_config.min_keypoint_confidence
            final_bbox_weight = confidence_config.bbox_weight
            final_keypoint_weight = confidence_config.keypoint_weight
            logger.info(
                f"Using confidence filtering from config: bbox >= {final_bbox_conf}, "
                f"keypoint >= {final_keypoint_conf}, weights: bbox={final_bbox_weight:.3f}, keypoint={final_keypoint_weight:.3f}"
            )
        else:
            # If config is disabled or not available, use function arguments or error if None
            if min_bbox_confidence is None or min_keypoint_confidence is None:
                raise ValueError(
                    "Confidence filtering parameters must be provided either via config file "
                    "(enabled confidence_filtering section) or CLI arguments"
                )

            if confidence_config:
                logger.info(
                    f"Config filtering disabled, using function arguments: bbox >= {final_bbox_conf}, keypoint >= {final_keypoint_conf}"
                )
            else:
                logger.info(
                    f"No confidence filtering config found, using function arguments: bbox >= {final_bbox_conf}, keypoint >= {final_keypoint_conf}"
                )

        pred_loader = PredictionLoader(
            pred_pkl_path,
            pipeline_config,
            min_bbox_confidence=final_bbox_conf,
            min_keypoint_confidence=final_keypoint_conf,
            bbox_weight=final_bbox_weight,
            keypoint_weight=final_keypoint_weight,
        )
        pred_keypoints, pred_bboxes, pred_scores = pred_loader.get_filtered_predictions(
            subject, action, camera_idx-1, original_video_path
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

        # Generate visualizations if requested
        if enable_visualization and visualization_output_dir is not None:
            try:
                logger.info(
                    f"Generating visualizations for {subject}/{action}/{camera_str}"
                )
                _generate_filtered_frames(
                    pred_keypoints=pred_keypoints[:min_len],
                    gt_keypoints=gt_keypoints[:min_len],
                    original_video_path=original_video_path,
                    output_dir=visualization_output_dir,
                    subject=subject,
                    action=safe_action_name,
                    camera_str=camera_str,
                    frame_step=frame_step,
                    visualize_gt=visualize_gt,
                    sync_offset=_get_sync_offset(
                        pipeline_config, subject, action, camera_idx-1
                    ),
                )
                logger.info(
                    f"Visualization completed for {subject}/{action}/{camera_str}"
                )
            except Exception as e:
                logger.error(f"Visualization failed for {pred_pkl_path}: {e}")

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
    enable_visualization=True,
    visualization_output_dir=None,
    frame_step=1,
    visualize_gt=True,
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
    gt_enum_class = import_class_from_string(
        pipeline_config.dataset.joint_enum_module)
    pred_enum_class = import_class_from_string(
        pipeline_config.dataset.keypoint_format)

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

    # Setup visualization if enabled
    if enable_visualization:
        if visualization_output_dir is None:
            visualization_output_dir = os.path.join(
                output_dir, "visualizations")
        os.makedirs(visualization_output_dir, exist_ok=True)
        logger.info(
            f"Visualization enabled. Output directory: {visualization_output_dir}"
        )
        logger.info(f"Frame step: {frame_step}, Include GT: {visualize_gt}")
    else:
        logger.info("Visualization disabled")

    pred_root = (
        pipeline_config.evaluation.input_dir or pipeline_config.detect.output_dir
    )
    # Pass the output path to the evaluator
    evaluator = MetricsEvaluator(output_path=output_dir)

    # Define the grouping keys for HumanEva
    grouping_keys = ["subject", "action", "camera"]

    # Create a wrapper function that includes the confidence and visualization parameters
    def data_loader_with_options(pred_pkl_path, pipeline_config, global_config):
        return humaneva_data_loader(
            pred_pkl_path,
            pipeline_config,
            global_config,
            min_bbox_confidence=min_bbox_confidence,
            min_keypoint_confidence=min_keypoint_confidence,
            enable_visualization=enable_visualization,
            visualization_output_dir=visualization_output_dir,
            frame_step=frame_step,
            visualize_gt=visualize_gt,
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

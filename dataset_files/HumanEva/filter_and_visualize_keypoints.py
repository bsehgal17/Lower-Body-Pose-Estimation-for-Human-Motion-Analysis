"""
HumanEva Keypoint Filtering and Visualization Script

This script filters predicted keypoints based on confidence thresholds and saves frames
with overlaid keypoints. It integrates with the HumanEva data loader pipeline and
properly handles synchronization data for frame alignment.

Features:
- Filters keypoints based on confidence thresholds
- Applies synchronization offsets for proper frame alignment
- Overlays filtered keypoints on video frames
- Saves annotated frames to output directory
- Supports both ground truth and prediction visualization
- Handles multiple filtering criteria (bbox confidence, keypoint confidence)
"""

import os
import cv2
import numpy as np
import logging
from typing import Tuple, Optional, Dict, Any
import pickle
import json

from config.pipeline_config import PipelineConfig
from config.global_config import GlobalConfig
from dataset_files.HumanEva.get_pred_keypoint import PredictionLoader
from dataset_files.HumanEva.humaneva_metadata import get_humaneva_metadata_from_video

logger = logging.getLogger(__name__)


class HumanEvaKeypointFilter:
    """
    Filters and visualizes HumanEva keypoints with frame synchronization support.
    """

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        global_config: GlobalConfig,
        min_bbox_confidence: float = 0.5,
        min_keypoint_confidence: float = 0.3,
        bbox_weight: float = 0.3,
        keypoint_weight: float = 0.7,
    ):
        self.pipeline_config = pipeline_config
        self.global_config = global_config
        self.min_bbox_confidence = min_bbox_confidence
        self.min_keypoint_confidence = min_keypoint_confidence
        self.bbox_weight = bbox_weight
        self.keypoint_weight = keypoint_weight

        # Color scheme for visualization
        self.colors = {
            "gt": (0, 255, 0),  # Green for ground truth
            "pred": (255, 0, 0),  # Blue for predictions
            "filtered": (0, 255, 255),  # Yellow for filtered predictions
            "skeleton": (255, 255, 255),  # White for skeleton connections
        }

        # Lower body skeleton connections (using standard pose estimation indices)
        self.skeleton_connections = [
            (11, 12),  # Left hip to right hip
            (11, 13),  # Left hip to left knee
            (13, 15),  # Left knee to left ankle
            (12, 14),  # Right hip to right knee
            (14, 16),  # Right knee to right ankle
        ]

    def _get_sync_offset(self, subject: str, action: str, camera_idx: int) -> int:
        """
        Get synchronization offset for proper frame alignment.

        Args:
            subject: Subject identifier (e.g., 'S1')
            action: Action name (e.g., 'Walking 1')
            camera_idx: Camera index (0, 1, 2)

        Returns:
            Frame offset for synchronization
        """
        try:
            if hasattr(self.pipeline_config, "dataset") and hasattr(
                self.pipeline_config.dataset, "sync_data"
            ):
                sync_start = self.pipeline_config.dataset.sync_data["data"][subject][
                    action
                ][camera_idx]
                return sync_start
        except (KeyError, AttributeError) as e:
            logger.warning(
                f"No sync data found for {subject}/{action}/camera_{camera_idx}: {e}"
            )

        return 0

    def _filter_keypoints_by_confidence(
        self, keypoints: np.ndarray, confidence_threshold: float
    ) -> np.ndarray:
        """
        Filter keypoints based on confidence scores.

        Args:
            keypoints: Array of shape [num_joints, 3] (x, y, confidence)
            confidence_threshold: Minimum confidence threshold

        Returns:
            Filtered keypoints with low-confidence points set to NaN
        """
        if keypoints is None or len(keypoints) == 0:
            return None

        filtered_keypoints = keypoints.copy()

        # Assume confidence is in the third column
        if keypoints.shape[1] >= 3:
            confidence_scores = keypoints[:, 2]
            low_confidence_mask = confidence_scores < confidence_threshold
            filtered_keypoints[low_confidence_mask, :] = np.nan

        # Check if we have enough valid keypoints
        valid_keypoints = ~np.isnan(filtered_keypoints[:, 0])
        if np.sum(valid_keypoints) < 3:
            logger.warning("Too few valid keypoints after filtering")
            return None

        return filtered_keypoints

    def _draw_keypoints_on_frame(
        self,
        frame: np.ndarray,
        keypoints: np.ndarray,
        color: Tuple[int, int, int],
        draw_skeleton: bool = True,
        point_radius: int = 4,
        line_thickness: int = 2,
    ) -> np.ndarray:
        """
        Draw keypoints and skeleton on frame.

        Args:
            frame: Input frame
            keypoints: Keypoints array [num_joints, 2 or 3]
            color: RGB color tuple
            draw_skeleton: Whether to draw skeleton connections
            point_radius: Radius of keypoint circles
            line_thickness: Thickness of skeleton lines

        Returns:
            Frame with keypoints drawn
        """
        if keypoints is None or len(keypoints) == 0:
            return frame

        annotated_frame = frame.copy()

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
            for start_idx, end_idx in self.skeleton_connections:
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
                    end_point = (int(keypoints[end_idx][0]), int(keypoints[end_idx][1]))
                    cv2.line(
                        annotated_frame, start_point, end_point, color, line_thickness
                    )

        return annotated_frame

    def process_video_with_keypoints(
        self,
        pred_pkl_path: str,
        output_dir: str,
        visualize_gt: bool = True,
        visualize_pred: bool = True,
        save_frames: bool = True,
        frame_range: Optional[Tuple[int, int]] = None,
        frame_step: int = 1,
    ) -> Dict[str, Any]:
        """
        Process a video with keypoint filtering and visualization.

        Args:
            pred_pkl_path: Path to prediction pickle file
            output_dir: Output directory for saved frames
            visualize_gt: Whether to visualize ground truth keypoints
            visualize_pred: Whether to visualize predicted keypoints
            save_frames: Whether to save annotated frames
            frame_range: Optional tuple (start_frame, end_frame) to limit processing
            frame_step: Step size for frame processing (1 = every frame, 2 = every other frame, etc.)

        Returns:
            Dictionary with processing statistics
        """
        try:
            # Extract metadata from prediction file
            json_path = pred_pkl_path.replace(".pkl", ".json")
            metadata = get_humaneva_metadata_from_video(json_path)
            if not metadata:
                raise ValueError(f"Could not parse metadata from {json_path}")

            subject, action, camera_str = (
                metadata["subject"],
                metadata["action"],
                metadata["camera"],
            )
            camera_idx = int(camera_str[1:]) - 1
            safe_action_name = action.replace(" ", "_")

            logger.info(f"Processing {subject}/{action}/{camera_str}")

            # Get sync offset for frame alignment
            sync_offset = self._get_sync_offset(subject, action, camera_idx)
            logger.info(f"Using sync offset: {sync_offset}")

            # Load predictions
            pred_loader = PredictionLoader(
                pred_pkl_path,
                self.pipeline_config,
                min_bbox_confidence=self.min_bbox_confidence,
                min_keypoint_confidence=self.min_keypoint_confidence,
                bbox_weight=self.bbox_weight,
                keypoint_weight=self.keypoint_weight,
            )

            # Get video path
            original_video_base = os.path.join(
                self.global_config.paths.input_dir, self.pipeline_config.paths.dataset
            )
            original_video_path = os.path.join(
                original_video_base,
                subject,
                "Image_Data",
                f"{safe_action_name}_({camera_str}).avi",
            )

            if not os.path.exists(original_video_path):
                raise FileNotFoundError(f"Video not found: {original_video_path}")

            # Load filtered predictions
            pred_keypoints, pred_bboxes, pred_scores = (
                pred_loader.get_filtered_predictions(
                    subject, action, camera_idx, original_video_path
                )
            )

            # Load ground truth if requested
            gt_keypoints = None
            if visualize_gt:
                csv_file_path = self.pipeline_config.paths.ground_truth_file
                gt_dir = os.path.dirname(csv_file_path)
                gt_pkl_name = f"{subject}_{safe_action_name}_{camera_str}_gt.pkl"
                gt_pkl_folder = os.path.join(gt_dir, "pickle_files")
                gt_pkl_path = os.path.join(gt_pkl_folder, gt_pkl_name)

                if os.path.exists(gt_pkl_path):
                    with open(gt_pkl_path, "rb") as f:
                        gt_data = pickle.load(f)
                    gt_keypoints = gt_data["keypoints"]
                    # Apply sync offset to GT as well
                    if sync_offset > 0 and len(gt_keypoints) > sync_offset:
                        gt_keypoints = gt_keypoints[sync_offset:]
                else:
                    logger.warning(f"GT pickle file not found: {gt_pkl_path}")

            # Create output directory
            video_output_dir = os.path.join(
                output_dir, f"{subject}_{safe_action_name}_{camera_str}_filtered"
            )
            os.makedirs(video_output_dir, exist_ok=True)

            # Open video
            cap = cv2.VideoCapture(original_video_path)
            if not cap.isOpened():
                raise IOError(f"Could not open video: {original_video_path}")

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            # Determine frame range
            if frame_range is None:
                start_frame = sync_offset
                end_frame = min(total_frames, len(pred_keypoints) + sync_offset)
            else:
                start_frame = max(frame_range[0], sync_offset)
                end_frame = min(
                    frame_range[1], total_frames, len(pred_keypoints) + sync_offset
                )

            logger.info(
                f"Processing frames {start_frame} to {end_frame} (step={frame_step})"
            )

            # Statistics
            stats = {
                "total_frames_processed": 0,
                "frames_with_valid_predictions": 0,
                "frames_with_gt": 0,
                "average_prediction_confidence": 0.0,
                "filtered_keypoints_count": 0,
            }

            processed_frames = 0
            confidence_sum = 0.0

            # Process frames
            for frame_idx in range(start_frame, end_frame, frame_step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    logger.warning(f"Could not read frame {frame_idx}")
                    continue

                # Calculate prediction index (accounting for sync offset)
                pred_idx = frame_idx - sync_offset

                annotated_frame = frame.copy()
                has_valid_predictions = False

                # Draw predictions
                if visualize_pred and 0 <= pred_idx < len(pred_keypoints):
                    pred_kpts = pred_keypoints[pred_idx]
                    if pred_kpts is not None:
                        # Additional filtering on the keypoints
                        filtered_pred_kpts = self._filter_keypoints_by_confidence(
                            pred_kpts, self.min_keypoint_confidence
                        )

                        if filtered_pred_kpts is not None:
                            annotated_frame = self._draw_keypoints_on_frame(
                                annotated_frame, filtered_pred_kpts, self.colors["pred"]
                            )
                            has_valid_predictions = True
                            stats["filtered_keypoints_count"] += np.sum(
                                ~np.isnan(filtered_pred_kpts[:, 0])
                            )

                            # Calculate average confidence
                            if filtered_pred_kpts.shape[1] >= 3:
                                valid_conf = filtered_pred_kpts[
                                    ~np.isnan(filtered_pred_kpts[:, 2]), 2
                                ]
                                if len(valid_conf) > 0:
                                    confidence_sum += np.mean(valid_conf)

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

                        annotated_frame = self._draw_keypoints_on_frame(
                            annotated_frame, gt_kpts, self.colors["gt"]
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

                # Save frame if requested
                if save_frames:
                    frame_filename = f"frame_{frame_idx:06d}.jpg"
                    frame_path = os.path.join(video_output_dir, frame_filename)
                    cv2.imwrite(frame_path, annotated_frame)

                if has_valid_predictions:
                    stats["frames_with_valid_predictions"] += 1

                processed_frames += 1
                stats["total_frames_processed"] += 1

                if processed_frames % 100 == 0:
                    logger.info(f"Processed {processed_frames} frames...")

            cap.release()

            # Finalize statistics
            if stats["frames_with_valid_predictions"] > 0:
                stats["average_prediction_confidence"] = (
                    confidence_sum / stats["frames_with_valid_predictions"]
                )

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
                "processing_params": {
                    "min_bbox_confidence": self.min_bbox_confidence,
                    "min_keypoint_confidence": self.min_keypoint_confidence,
                    "bbox_weight": self.bbox_weight,
                    "keypoint_weight": self.keypoint_weight,
                    "frame_range": [start_frame, end_frame],
                    "frame_step": frame_step,
                },
                "statistics": stats,
            }

            with open(summary_path, "w") as f:
                json.dump(summary_data, f, indent=2)

            logger.info(f"Processing completed. Output saved to: {video_output_dir}")
            logger.info(f"Statistics: {stats}")

            return summary_data

        except Exception as e:
            logger.error(f"Error processing video {pred_pkl_path}: {e}")
            raise


def filter_and_visualize_humaneva_data(
    pred_pkl_path: str,
    pipeline_config: PipelineConfig,
    global_config: GlobalConfig,
    output_dir: str,
    min_bbox_confidence: float = 0.5,
    min_keypoint_confidence: float = 0.3,
    bbox_weight: float = 0.3,
    keypoint_weight: float = 0.7,
    visualize_gt: bool = True,
    visualize_pred: bool = True,
    frame_range: Optional[Tuple[int, int]] = None,
    frame_step: int = 1,
) -> Dict[str, Any]:
    """
    Main function to filter and visualize HumanEva keypoints.

    This function can be called from the HumanEva data loader to generate
    visualized frames with filtered keypoints.

    Args:
        pred_pkl_path: Path to prediction pickle file
        pipeline_config: Pipeline configuration
        global_config: Global configuration
        output_dir: Output directory for frames
        min_bbox_confidence: Minimum bounding box confidence threshold
        min_keypoint_confidence: Minimum keypoint confidence threshold
        bbox_weight: Weight for bbox confidence in overall score
        keypoint_weight: Weight for keypoint confidence in overall score
        visualize_gt: Whether to show ground truth keypoints
        visualize_pred: Whether to show predicted keypoints
        frame_range: Optional frame range to process
        frame_step: Step size for frame processing

    Returns:
        Processing summary dictionary
    """
    filter_instance = HumanEvaKeypointFilter(
        pipeline_config=pipeline_config,
        global_config=global_config,
        min_bbox_confidence=min_bbox_confidence,
        min_keypoint_confidence=min_keypoint_confidence,
        bbox_weight=bbox_weight,
        keypoint_weight=keypoint_weight,
    )

    return filter_instance.process_video_with_keypoints(
        pred_pkl_path=pred_pkl_path,
        output_dir=output_dir,
        visualize_gt=visualize_gt,
        visualize_pred=visualize_pred,
        save_frames=True,
        frame_range=frame_range,
        frame_step=frame_step,
    )


if __name__ == "__main__":
    # Example usage for testing
    import argparse
    from config.global_config import GlobalConfig
    from config.pipeline_config import PipelineConfig

    parser = argparse.ArgumentParser(
        description="Filter and visualize HumanEva keypoints"
    )
    parser.add_argument(
        "--pred_pkl", required=True, help="Path to prediction pickle file"
    )
    parser.add_argument("--config", required=True, help="Path to pipeline config YAML")
    parser.add_argument(
        "--output_dir", required=True, help="Output directory for frames"
    )
    parser.add_argument(
        "--min_bbox_conf", type=float, default=0.5, help="Min bbox confidence"
    )
    parser.add_argument(
        "--min_keypoint_conf", type=float, default=0.3, help="Min keypoint confidence"
    )
    parser.add_argument("--frame_step", type=int, default=1, help="Frame step size")
    parser.add_argument(
        "--no_gt", action="store_true", help="Don't visualize ground truth"
    )

    args = parser.parse_args()

    # Load configurations
    global_config = GlobalConfig()
    pipeline_config = PipelineConfig.from_yaml(args.config)

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Run filtering and visualization
    result = filter_and_visualize_humaneva_data(
        pred_pkl_path=args.pred_pkl,
        pipeline_config=pipeline_config,
        global_config=global_config,
        output_dir=args.output_dir,
        min_bbox_confidence=args.min_bbox_conf,
        min_keypoint_confidence=args.min_keypoint_conf,
        visualize_gt=not args.no_gt,
        frame_step=args.frame_step,
    )

    print(f"Processing completed. Summary: {result}")

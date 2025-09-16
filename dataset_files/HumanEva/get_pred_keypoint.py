import os
import pickle
import logging
import numpy as np
from utils.video_io import get_video_resolution, rescale_keypoints

logger = logging.getLogger(__name__)


class PredictionLoader:
    """
    Handles loading and preprocessing of prediction data for HumanEva dataset.
    """

    def __init__(
        self,
        pred_pkl_path,
        pipeline_config=None,
        min_bbox_confidence=0.3,
        min_keypoint_confidence=0.2,
        bbox_weight=0.3,
        keypoint_weight=0.7,
    ):
        """
        Initialize the PredictionLoader.

        Args:
            pred_pkl_path (str): Path to the prediction pickle file
            pipeline_config: PipelineConfig object for sync data access
            min_bbox_confidence (float): Minimum bounding box confidence threshold
            min_keypoint_confidence (float): Minimum keypoint confidence threshold
            bbox_weight (float): Weight for bbox confidence in overall score calculation
            keypoint_weight (float): Weight for keypoint confidence in overall score calculation
        """
        self.pred_pkl_path = pred_pkl_path
        self.pipeline_config = pipeline_config
        self.min_bbox_confidence = min_bbox_confidence
        self.min_keypoint_confidence = min_keypoint_confidence
        self.bbox_weight = bbox_weight
        self.keypoint_weight = keypoint_weight

    def load_raw_predictions(self):
        """
        Load raw prediction data from pickle file.

        Returns:
            dict: Raw prediction data loaded from pickle file
        """
        try:
            with open(self.pred_pkl_path, "rb") as f:
                pred_data = pickle.load(f)
            return pred_data
        except Exception as e:
            logger.error(f"Error loading predictions from {self.pred_pkl_path}: {e}")
            raise

    def _calculate_skeleton_confidence(self, keypoints, bbox_score=None):
        """
        Calculate overall confidence score for a skeleton.

        Args:
            keypoints (np.ndarray): Keypoints array with shape [N, 3] (x, y, confidence)
            bbox_score (float): Bounding box confidence score

        Returns:
            dict: Confidence metrics including overall_score, valid_keypoints_count, avg_keypoint_confidence
        """
        if keypoints is None or len(keypoints) == 0:
            return {
                "overall_score": 0.0,
                "valid_keypoints_count": 0,
                "avg_keypoint_confidence": 0.0,
                "bbox_score": bbox_score if bbox_score is not None else 0.0,
            }

        # Extract confidence scores (3rd column)
        if keypoints.shape[1] >= 3:
            confidence_scores = keypoints[:, 2]
        else:
            # If no confidence scores available, assume all keypoints are valid
            confidence_scores = np.ones(len(keypoints))

        # Count keypoints above threshold
        valid_keypoints = confidence_scores >= self.min_keypoint_confidence
        valid_keypoints_count = np.sum(valid_keypoints)

        # Calculate average confidence of valid keypoints
        if valid_keypoints_count > 0:
            avg_keypoint_confidence = np.mean(confidence_scores[valid_keypoints])
        else:
            avg_keypoint_confidence = 0.0

        # Calculate overall score combining bbox and keypoint confidences
        bbox_contribution = (
            bbox_score if bbox_score is not None else 0.5
        ) * self.bbox_weight
        keypoint_contribution = avg_keypoint_confidence * self.keypoint_weight

        overall_score = bbox_contribution + keypoint_contribution

        return {
            "overall_score": overall_score,
            "valid_keypoints_count": valid_keypoints_count,
            "avg_keypoint_confidence": avg_keypoint_confidence,
            "bbox_score": bbox_score if bbox_score is not None else 0.0,
        }

    def _select_best_skeleton(self, people_data):
        """
        Select the best skeleton from multiple detections based on overall confidence.

        Args:
            people_data (list): List of person detection dictionaries

        Returns:
            dict or None: Best person data or None if no valid skeleton found
        """
        if not people_data:
            return None

        valid_skeletons = []

        for person in people_data:
            # Extract data
            kpts_raw = person.get("keypoints", [])
            bbox = person.get("bboxes", [0, 0, 0, 0])
            bbox_score = person.get("scores", 0.0)

            # Handle nested list structure for keypoints
            if (
                isinstance(kpts_raw, list)
                and len(kpts_raw) > 0
                and isinstance(kpts_raw[0], np.ndarray)
            ):
                kpts = kpts_raw[0]
            else:
                kpts = np.array(kpts_raw)

            # Handle any extra dimensions on the resulting array
            if kpts.ndim == 4:
                kpts = np.squeeze(kpts, axis=1)
            elif kpts.ndim == 3 and kpts.shape[0] == 1:
                kpts = np.squeeze(kpts, axis=0)

            # Filter by bbox confidence first
            if bbox_score < self.min_bbox_confidence:
                continue

            # Calculate skeleton confidence metrics
            confidence_metrics = self._calculate_skeleton_confidence(kpts, bbox_score)

            # Filter by minimum valid keypoints (at least 3 for basic pose)
            if confidence_metrics["valid_keypoints_count"] < 3:
                continue

            valid_skeletons.append(
                {
                    "person_data": person,
                    "keypoints": kpts,
                    "bbox": bbox,
                    "bbox_score": bbox_score,
                    "confidence_metrics": confidence_metrics,
                }
            )

        if not valid_skeletons:
            return None

        # Select skeleton with best overall confidence score
        best_skeleton = max(
            valid_skeletons, key=lambda x: x["confidence_metrics"]["overall_score"]
        )

        return best_skeleton["person_data"]

    def extract_keypoints_data(self, pred_data):
        """
        Extract keypoints, bboxes, and scores from raw prediction data.
        Now includes filtering for multiple skeletons based on confidence scores.

        Args:
            pred_data (dict): Raw prediction data

        Returns:
            tuple: (pred_keypoints, pred_bboxes, pred_scores) lists
        """
        pred_keypoints = []
        pred_bboxes = []
        pred_scores = []

        for frame_idx, frame in enumerate(pred_data["keypoints"]):
            people = frame["keypoints"]
            if not people:
                # If no predictions, append placeholders to keep lists aligned
                pred_keypoints.append(None)
                pred_bboxes.append(None)
                pred_scores.append(None)
                continue

            # Select best skeleton from multiple detections using confidence filtering
            best_person = self._select_best_skeleton(people)

            if best_person is None:
                # No skeleton meets confidence criteria
                logger.debug(
                    f"Frame {frame_idx}: No skeleton meets confidence criteria"
                )
                pred_keypoints.append(None)
                pred_bboxes.append(None)
                pred_scores.append(None)
                continue

            # Extract keypoints, bbox, and score from best skeleton
            kpts_raw = best_person.get("keypoints", [])

            # Handle nested list structure for keypoints
            if (
                isinstance(kpts_raw, list)
                and len(kpts_raw) > 0
                and isinstance(kpts_raw[0], np.ndarray)
            ):
                kpts = kpts_raw[0]
            else:
                kpts = np.array(kpts_raw)

            # Handle any extra dimensions on the resulting array
            if kpts.ndim == 4:
                kpts = np.squeeze(kpts, axis=1)
            elif kpts.ndim == 3 and kpts.shape[0] == 1:
                kpts = np.squeeze(kpts, axis=0)

            bbox = best_person.get("bboxes", [0, 0, 0, 0])
            score = best_person.get("scores", 0.0)

            pred_keypoints.append(kpts)
            pred_bboxes.append(bbox)
            pred_scores.append(score)

        # Log filtering statistics
        valid_frames = sum(1 for kpts in pred_keypoints if kpts is not None)
        total_frames = len(pred_keypoints)
        logger.info(
            f"Skeleton filtering results: {valid_frames}/{total_frames} frames with valid skeletons "
            f"(bbox_conf >= {self.min_bbox_confidence}, kpt_conf >= {self.min_keypoint_confidence})"
        )

        return pred_keypoints, pred_bboxes, pred_scores

    def set_confidence_thresholds(
        self, min_bbox_confidence=None, min_keypoint_confidence=None
    ):
        """
        Update confidence thresholds for skeleton filtering.

        Args:
            min_bbox_confidence (float): Minimum bounding box confidence threshold
            min_keypoint_confidence (float): Minimum keypoint confidence threshold
        """
        if min_bbox_confidence is not None:
            self.min_bbox_confidence = min_bbox_confidence
            logger.info(f"Updated bbox confidence threshold to {min_bbox_confidence}")

        if min_keypoint_confidence is not None:
            self.min_keypoint_confidence = min_keypoint_confidence
            logger.info(
                f"Updated keypoint confidence threshold to {min_keypoint_confidence}"
            )

    def set_confidence_weights(self, bbox_weight=None, keypoint_weight=None):
        """
        Update confidence weights for overall score calculation.

        Args:
            bbox_weight (float): Weight for bbox confidence in overall score calculation
            keypoint_weight (float): Weight for keypoint confidence in overall score calculation
        """
        if bbox_weight is not None:
            self.bbox_weight = bbox_weight
            logger.info(f"Updated bbox weight to {bbox_weight}")

        if keypoint_weight is not None:
            self.keypoint_weight = keypoint_weight
            logger.info(f"Updated keypoint weight to {keypoint_weight}")

        # Normalize weights to ensure they sum to 1.0
        total_weight = self.bbox_weight + self.keypoint_weight
        if total_weight != 1.0:
            self.bbox_weight = self.bbox_weight / total_weight
            self.keypoint_weight = self.keypoint_weight / total_weight
            logger.info(
                f"Normalized weights: bbox={self.bbox_weight:.3f}, keypoint={self.keypoint_weight:.3f}"
            )

    def get_filtering_stats(self, pred_data):
        """
        Get statistics about skeleton filtering without processing the data.

        Args:
            pred_data (dict): Raw prediction data

        Returns:
            dict: Statistics about filtering results
        """
        stats = {
            "total_frames": 0,
            "frames_with_detections": 0,
            "frames_with_multiple_skeletons": 0,
            "frames_after_filtering": 0,
            "avg_skeletons_per_frame": 0.0,
        }

        total_detections = 0

        for frame in pred_data["keypoints"]:
            people = frame["keypoints"]
            stats["total_frames"] += 1

            if people:
                stats["frames_with_detections"] += 1
                total_detections += len(people)

                if len(people) > 1:
                    stats["frames_with_multiple_skeletons"] += 1

                # Check if any skeleton would pass filtering
                best_person = self._select_best_skeleton(people)
                if best_person is not None:
                    stats["frames_after_filtering"] += 1

        if stats["frames_with_detections"] > 0:
            stats["avg_skeletons_per_frame"] = (
                total_detections / stats["frames_with_detections"]
            )

        return stats

    def rescale_predictions(self, pred_keypoints, pred_bboxes, original_video_path):
        """
        Rescale keypoints and bboxes if video resolutions differ.

        Args:
            pred_keypoints (list): List of keypoint arrays
            pred_bboxes (list): List of bounding boxes
            original_video_path (str): Path to original video file

        Returns:
            tuple: (rescaled_keypoints, rescaled_bboxes)
        """
        # Get original video resolution
        orig_w, orig_h = get_video_resolution(original_video_path)

        # Get prediction video resolution
        pred_video_path = os.path.join(
            os.path.dirname(self.pred_pkl_path),
            f"{os.path.splitext(os.path.basename(self.pred_pkl_path))[0]}.avi",
        )

        if not os.path.exists(pred_video_path):
            logger.warning(f"Prediction video not found: {pred_video_path}")
            return pred_keypoints, pred_bboxes

        test_w, test_h = get_video_resolution(pred_video_path)

        if (test_w, test_h) == (orig_w, orig_h):
            # No rescaling needed
            return pred_keypoints, pred_bboxes

        # Calculate scaling factors
        scale_x = orig_w / test_w
        scale_y = orig_h / test_h

        logger.info(
            f"Rescaling predictions: ({test_w}x{test_h}) -> ({orig_w}x{orig_h})"
        )

        # Rescale keypoints
        rescaled_keypoints = []
        for kpts in pred_keypoints:
            if kpts is not None:
                rescaled_kpts = rescale_keypoints(kpts, scale_x, scale_y)
                rescaled_keypoints.append(rescaled_kpts)
            else:
                rescaled_keypoints.append(None)

        # Rescale bboxes
        rescaled_bboxes = []
        for bbox in pred_bboxes:
            if bbox is not None:
                # Unpack the nested lists into individual coordinates
                coords = bbox[0] if isinstance(bbox[0], list) else bbox

                # Apply scaling to each coordinate
                rescaled_bbox = [
                    coords[0] * scale_x,
                    coords[1] * scale_y,
                    coords[2] * scale_x,
                    coords[3] * scale_y,
                ]
                rescaled_bboxes.append(rescaled_bbox)
            else:
                rescaled_bboxes.append(None)

        return rescaled_keypoints, rescaled_bboxes

    def apply_synchronization(
        self, pred_keypoints, pred_bboxes, pred_scores, subject, action, camera_idx
    ):
        """
        Apply synchronization offset to prediction data.

        Args:
            pred_keypoints (list): List of keypoint arrays
            pred_bboxes (list): List of bounding boxes
            pred_scores (list): List of confidence scores
            subject (str): Subject identifier
            action (str): Action name
            camera_idx (int): Camera index

        Returns:
            tuple: (synced_keypoints, synced_bboxes, synced_scores)
        """
        if not self.pipeline_config or not hasattr(
            self.pipeline_config.dataset, "sync_data"
        ):
            logger.warning("No sync data available, using offset 0")
            return pred_keypoints, pred_bboxes, pred_scores

        try:
            sync_start = self.pipeline_config.dataset.sync_data["data"][subject][
                action
            ][camera_idx]

            if sync_start >= len(pred_keypoints):
                logger.warning(
                    f"Sync index {sync_start} exceeds prediction length {len(pred_keypoints)}"
                )
                return pred_keypoints, pred_bboxes, pred_scores

        except KeyError:
            logger.warning(
                f"No sync index for {subject} | {action} | camera {camera_idx}"
            )
            sync_start = 0

        # Apply synchronization offset
        synced_keypoints = pred_keypoints[sync_start:]
        synced_bboxes = pred_bboxes[sync_start:]
        synced_scores = pred_scores[sync_start:]

        return synced_keypoints, synced_bboxes, synced_scores

    def get_predictions(self, subject, action, camera_idx, original_video_path):
        """
        Main method to get processed prediction data.

        Args:
            subject (str): Subject identifier
            action (str): Action name
            camera_idx (int): Camera index
            original_video_path (str): Path to original video file

        Returns:
            tuple: (pred_keypoints, pred_bboxes, pred_scores) processed lists
        """
        # Load raw predictions
        pred_data = self.load_raw_predictions()

        # Extract keypoints data
        pred_keypoints, pred_bboxes, pred_scores = self.extract_keypoints_data(
            pred_data
        )

        # Rescale if needed
        pred_keypoints, pred_bboxes = self.rescale_predictions(
            pred_keypoints, pred_bboxes, original_video_path
        )

        # Apply synchronization
        pred_keypoints, pred_bboxes, pred_scores = self.apply_synchronization(
            pred_keypoints, pred_bboxes, pred_scores, subject, action, camera_idx
        )

        return pred_keypoints, pred_bboxes, pred_scores


# Example usage with enhanced confidence filtering:
"""
# Basic usage with default thresholds and weights
loader = PredictionLoader("path/to/predictions.pkl")

# Usage with custom confidence thresholds and weights
loader = PredictionLoader(
    "path/to/predictions.pkl",
    min_bbox_confidence=0.5,      # Higher bbox confidence threshold
    min_keypoint_confidence=0.3,  # Higher keypoint confidence threshold
    bbox_weight=0.4,              # Custom bbox weight (40%)
    keypoint_weight=0.6,          # Custom keypoint weight (60%)
)

# Get filtering statistics before processing
raw_data = loader.load_raw_predictions()
stats = loader.get_filtering_stats(raw_data)
print(f"Filtering stats: {stats}")

# Process with filtering
pred_keypoints, pred_bboxes, pred_scores = loader.extract_keypoints_data(raw_data)

# Adjust thresholds and weights on the fly if needed
loader.set_confidence_thresholds(min_bbox_confidence=0.4, min_keypoint_confidence=0.25)
loader.set_confidence_weights(bbox_weight=0.2, keypoint_weight=0.8)  # Prioritize keypoints more
"""

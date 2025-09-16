import os
import pickle
import logging
import numpy as np
from utils.video_io import get_video_resolution, rescale_keypoints

logger = logging.getLogger(__name__)


class PredictionLoader:
    """
    Loads and preprocesses person-centric prediction data from HumanEva dataset.
    Filters skeletons and keypoints based on confidence thresholds.
    """

    def __init__(
        self,
        pred_pkl_path,
        pipeline_config,
        min_bbox_confidence,
        min_keypoint_confidence,
        bbox_weight,
        keypoint_weight,
    ):
        self.pred_pkl_path = pred_pkl_path
        self.pipeline_config = pipeline_config
        self.min_bbox_confidence = min_bbox_confidence
        self.min_keypoint_confidence = min_keypoint_confidence
        self.bbox_weight = bbox_weight
        self.keypoint_weight = keypoint_weight

    def load_raw_predictions(self):
        """Load raw prediction pickle file."""
        with open(self.pred_pkl_path, "rb") as f:
            pred_data = pickle.load(f)
        return pred_data

    def _calculate_skeleton_confidence(self, keypoints, bbox_score):
        """Compute overall skeleton confidence score."""
        if keypoints is None or len(keypoints) == 0:
            return {"overall_score": 0.0, "valid_keypoints_count": 0,
                    "avg_keypoint_confidence": 0.0, "bbox_score": bbox_score}

        confidence_scores = keypoints[:, 2] if keypoints.shape[1] >= 3 else np.ones(
            len(keypoints))
        valid_keypoints = confidence_scores >= self.min_keypoint_confidence
        valid_count = np.sum(valid_keypoints)
        avg_conf = np.mean(
            confidence_scores[valid_keypoints]) if valid_count > 0 else 0.0

        overall_score = bbox_score * self.bbox_weight + avg_conf * self.keypoint_weight
        return {"overall_score": overall_score, "valid_keypoints_count": valid_count,
                "avg_keypoint_confidence": avg_conf, "bbox_score": bbox_score}

    def _select_best_skeleton(self, people_data):
        """Select the best skeleton in a frame based on confidence score."""
        valid_skeletons = []
        for person in people_data:
            kpts_raw = person.get("keypoints", [])
            bbox_score = person.get("scores")
            kpts = np.array(kpts_raw[0]) if isinstance(
                kpts_raw[0], list) else np.array(kpts_raw)
            if kpts.ndim == 4:
                kpts = np.squeeze(kpts, axis=1)
            if kpts.ndim == 3 and kpts.shape[0] == 1:
                kpts = np.squeeze(kpts, axis=0)

            if bbox_score < self.min_bbox_confidence:
                continue
            conf_metrics = self._calculate_skeleton_confidence(
                kpts, bbox_score)
            if conf_metrics["valid_keypoints_count"] < 3:
                continue

            valid_skeletons.append({"person_data": person, "keypoints": kpts,
                                    "confidence_metrics": conf_metrics})

        if not valid_skeletons:
            return None
        best_skeleton = max(
            valid_skeletons, key=lambda x: x["confidence_metrics"]["overall_score"])
        return best_skeleton["person_data"]

    def _filter_keypoints(self, kpts, visibility):
        """
        Filter keypoints based on visibility only.
        Set low-visibility keypoints to NaN.
        Discard skeleton if fewer than 3 visible joints.

        Args:
            kpts (np.ndarray): keypoints array of shape [num_joints, 3] (x, y, _)
            visibility (np.ndarray or list): visibility flags for each keypoint, shape [num_joints]

        Returns:
            np.ndarray or None: filtered keypoints or None if too few valid joints
        """
        visibility = np.array(visibility).reshape(-1)
        kpts_filtered = kpts.copy()

        # Set joints below threshold to NaN
        low_vis_mask = visibility < self.min_keypoint_confidence  # visibility threshold
        kpts_filtered[low_vis_mask, :] = np.nan

        # Discard skeleton if fewer than 3 visible joints
        if np.sum(~low_vis_mask) < 3:
            return None

        return kpts_filtered

    def _extract_keypoints_data(self, pred_data):
        """Extract keypoints, bboxes, and scores from person-centric predictions."""
        if "persons" not in pred_data:
            raise ValueError("Prediction data does not contain 'persons' key")

        max_frame_idx = max(
            pose["frame_idx"] for person in pred_data["persons"] for pose in person["poses"])
        total_frames = max_frame_idx + 1

        pred_keypoints = [None] * total_frames
        pred_bboxes = [None] * total_frames
        pred_scores = [None] * total_frames
        frames_data = {frame_idx: [] for frame_idx in range(total_frames)}

        for person in pred_data["persons"]:
            person_id = person["person_id"]
            detections_by_frame = {det["frame_idx"]
                : det for det in person["detections"]}
            for pose in person["poses"]:
                frame_idx = pose["frame_idx"]
                detection = detections_by_frame.get(frame_idx)
                person_data = {
                    "keypoints": pose["keypoints"],
                    "bboxes": pose["bbox"],
                    "scores": detection.get("score") if detection else pose.get("bbox_scores", [0])[0],
                    "person_id": person_id,
                    "keypoints_visible": pose.get("keypoints_visible")
                }
                frames_data[frame_idx].append(person_data)

        for frame_idx, people in frames_data.items():
            if not people:
                continue
            best_person = self._select_best_skeleton(people)
            if best_person is None:
                continue

            kpts_raw = best_person.get("keypoints")
            kpts = np.array(kpts_raw[0]) if isinstance(
                kpts_raw[0], list) else np.array(kpts_raw)

            # Get visibility array
            visibility = best_person.get(
                "keypoints_visible", np.ones(kpts.shape[0]))

            # Filter keypoints based on visibility
            kpts_filtered = self._filter_keypoints(kpts, visibility)
            if kpts_filtered is None:
                continue

            pred_keypoints[frame_idx] = kpts_filtered
            pred_bboxes[frame_idx] = best_person.get("bboxes", [0, 0, 0, 0])
            pred_scores[frame_idx] = best_person.get("scores", 0.0)

        return pred_keypoints, pred_bboxes, pred_scores

    def _rescale_predictions(self, pred_keypoints, pred_bboxes, original_video_path):
        orig_w, orig_h = get_video_resolution(original_video_path)
        pred_video_path = os.path.join(
            os.path.dirname(self.pred_pkl_path),
            f"{os.path.splitext(os.path.basename(self.pred_pkl_path))[0]}.avi",
        )

        if not os.path.exists(pred_video_path):
            logger.warning(f"Prediction video not found: {pred_video_path}")
            return pred_keypoints, pred_bboxes

        test_w, test_h = get_video_resolution(pred_video_path)
        if (test_w, test_h) == (orig_w, orig_h):
            return pred_keypoints, pred_bboxes

        scale_x = orig_w / test_w
        scale_y = orig_h / test_h

        rescaled_keypoints = [rescale_keypoints(
            k, scale_x, scale_y) if k is not None else None for k in pred_keypoints]
        rescaled_bboxes = [
            [b[0]*scale_x, b[1]*scale_y, b[2]*scale_x,
                b[3]*scale_y] if b is not None else None
            for b in pred_bboxes
        ]

        return rescaled_keypoints, rescaled_bboxes

    def _apply_synchronization(self, pred_keypoints, pred_bboxes, pred_scores, subject, action, camera_idx):
        if not self.pipeline_config or not hasattr(self.pipeline_config.dataset, "sync_data"):
            return pred_keypoints, pred_bboxes, pred_scores

        try:
            sync_start = self.pipeline_config.dataset.sync_data["data"][subject][action][camera_idx]
            if sync_start >= len(pred_keypoints):
                sync_start = 0
        except KeyError:
            sync_start = 0

        return pred_keypoints[sync_start:], pred_bboxes[sync_start:], pred_scores[sync_start:]

    # --- Single function for everything ---
    def get_filtered_predictions(self, subject, action, camera_idx, original_video_path):
        """
        Load, filter, rescale, and synchronize predictions in one call.
        """
        pred_data = self.load_raw_predictions()
        pred_keypoints, pred_bboxes, pred_scores = self._extract_keypoints_data(
            pred_data)
        pred_keypoints, pred_bboxes = self._rescale_predictions(
            pred_keypoints, pred_bboxes, original_video_path)
        pred_keypoints, pred_bboxes, pred_scores = self._apply_synchronization(
            pred_keypoints, pred_bboxes, pred_scores, subject, action, camera_idx
        )
        return pred_keypoints, pred_bboxes, pred_scores

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
    ):
        self.pred_pkl_path = pred_pkl_path
        self.pipeline_config = pipeline_config

    def load_raw_predictions(self):
        """Load raw prediction pickle file and convert to dictionary format."""
        with open(self.pred_pkl_path, "rb") as f:
            pred_data = pickle.load(f)

        # Handle both SavedData objects and legacy dictionaries
        if hasattr(pred_data, "to_dict"):
            # It's a SavedData object, convert to dictionary
            pred_data = pred_data.to_dict()
        elif hasattr(pred_data, "video_data"):
            # It's a SavedData object without to_dict method (older version)
            pred_data = pred_data.video_data.to_dict()

        return pred_data

    def _filter_keypoints(self, kpts, visibility):
        """
        Bypass keypoint confidence filtering: return all keypoints as-is.
        """
        return kpts

    def _extract_keypoints_data(self, pred_data):
        """Extract keypoints, bboxes, and scores from person-centric predictions."""
        if "persons" not in pred_data:
            raise ValueError("Prediction data does not contain 'persons' key")

        max_frame_idx = max(
            pose["frame_idx"]
            for person in pred_data["persons"]
            for pose in person["poses"]
        )
        total_frames = max_frame_idx + 1

        pred_keypoints = [None] * total_frames
        pred_bboxes = [None] * total_frames
        pred_scores = [None] * total_frames
        frames_data = {frame_idx: [] for frame_idx in range(total_frames)}

        for person in pred_data["persons"]:
            person_id = person["person_id"]
            detections_by_frame = {
                det["frame_idx"]: det for det in person["detections"]
            }
            for pose in person["poses"]:
                frame_idx = pose["frame_idx"]
                detection = detections_by_frame.get(frame_idx)
                person_data = {
                    "keypoints": pose["keypoints"],
                    "bboxes": pose["bbox"],
                    "scores": detection.get("score")
                    if detection
                    else pose.get("bbox_scores", [0])[0],
                    "person_id": person_id,
                    "keypoints_visible": pose.get("keypoints_visible"),
                }
                frames_data[frame_idx].append(person_data)

        for frame_idx, people in frames_data.items():
            if not people:
                continue  # skip frames with no people

            # Just take the first person in the frame
            person = people[0]

            kpts_raw = person.get("keypoints")
            if kpts_raw is None:
                continue  # skip if keypoints are missing

            # Convert keypoints to np.array safely
            try:
                kpts = (
                    np.array(kpts_raw)
                    if isinstance(kpts_raw, list)
                    else np.array(kpts_raw)
                )
            except Exception:
                continue  # skip if keypoints are malformed

            visibility = person.get("keypoints_visible")
            if visibility is None or len(visibility) != kpts.shape[0]:
                visibility = np.ones(kpts.shape[0])  # default visibility

            # Filter keypoints
            kpts_filtered = self._filter_keypoints(kpts, visibility)
            if kpts_filtered is None:
                continue  # skip if filtering fails

            # Safe extraction of bbox and scores
            pred_keypoints[frame_idx] = kpts_filtered
            pred_bboxes[frame_idx] = person.get("bboxes", [0, 0, 0, 0])
            pred_scores[frame_idx] = person.get("scores", 0.0)

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

        rescaled_keypoints = [
            rescale_keypoints(k, scale_x, scale_y) if k is not None else None
            for k in pred_keypoints
        ]
        rescaled_bboxes = [
            [b[0] * scale_x, b[1] * scale_y, b[2] * scale_x, b[3] * scale_y]
            if b is not None
            else None
            for b in pred_bboxes
        ]

        return rescaled_keypoints, rescaled_bboxes

    def _apply_synchronization(
        self, pred_keypoints, pred_bboxes, pred_scores, subject, action, camera_idx
    ):
        if not self.pipeline_config or not hasattr(
            self.pipeline_config.dataset, "sync_data"
        ):
            return pred_keypoints, pred_bboxes, pred_scores

        try:
            sync_start = self.pipeline_config.dataset.sync_data["data"][subject][
                action
            ][camera_idx]
            if sync_start >= len(pred_keypoints):
                sync_start = 0
        except KeyError:
            sync_start = 0

        return (
            pred_keypoints[sync_start:],
            pred_bboxes[sync_start:],
            pred_scores[sync_start:],
        )

    # --- Single function for everything ---
    def get_filtered_predictions(
        self, subject, action, camera_idx, original_video_path
    ):
        """
        Load, filter, rescale, and synchronize predictions in one call.
        """
        pred_data = self.load_raw_predictions()
        pred_keypoints, pred_bboxes, pred_scores = self._extract_keypoints_data(
            pred_data
        )
        pred_keypoints, pred_bboxes = self._rescale_predictions(
            pred_keypoints, pred_bboxes, original_video_path
        )
        pred_keypoints, pred_bboxes, pred_scores = self._apply_synchronization(
            pred_keypoints, pred_bboxes, pred_scores, subject, action, camera_idx
        )
        return pred_keypoints, pred_bboxes, pred_scores

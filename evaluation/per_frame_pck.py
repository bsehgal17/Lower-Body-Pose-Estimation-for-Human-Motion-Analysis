from evaluation.base_calculators import BasePCKCalculator, average_if_tuple
from utils.pck_utils import compute_norm_length
from utils.pck_utils import select_norm_joints
import numpy as np
import logging

logger = logging.getLogger(__name__)


class PerFramePCKCalculator(BasePCKCalculator):
    def __init__(self, params, gt_enum, pred_enum, verbose=False):
        if "threshold" not in params:
            raise ValueError(
                "Parameter 'threshold' is required for PerFramePCKCalculator."
            )

        threshold = params["threshold"]
        joints_to_evaluate = params.get("joints_to_evaluate", None)
        norm_joints = params.get("norm_joints", select_norm_joints(joints_to_evaluate))
        if verbose:
            print(f"[PerFramePCKCalculator] Using normalization joints: {norm_joints}")
        super().__init__(threshold=threshold, joints_to_evaluate=joints_to_evaluate)
        if gt_enum is None or pred_enum is None:
            raise ValueError("Both gt_enum and pred_enum must be provided.")
        self.gt_enum = gt_enum
        self.pred_enum = pred_enum
        self.norm_joints = norm_joints

    # auto_select_norm_joints removed; use select_norm_joints from utils.pck_utils instead

    def compute(self, gt_keypoints, pred_keypoints):
        gt, pred = np.array(gt_keypoints), np.array(pred_keypoints)
        if pred.shape[1] == 1:
            pred = pred.squeeze(1)

        if gt.shape[0] != pred.shape[0] or gt.ndim != 3:
            raise ValueError("Input shape mismatch")

        # Compute normalization length using shared utility
        gt_keypoints_proc = np.array(
            [[average_if_tuple(x) for x in frame] for frame in gt]
        )
        norm_length = compute_norm_length(
            self.gt_enum, self.norm_joints, gt_keypoints_proc
        )

        if self.joints_to_evaluate is None:
            self.joints_to_evaluate = [j.name for j in self.gt_enum]

        gt_pts, pred_pts = [], []

        for joint in self.joints_to_evaluate:
            if joint not in self.gt_enum.__members__:
                logger.warning(f"Joint '{joint}' not found in GT enum. Skipping.")
                continue
            if joint not in self.pred_enum.__members__:
                logger.warning(f"Joint '{joint}' not found in Pred enum. Skipping.")
                continue

            g_idx, p_idx = self.gt_enum[joint].value, self.pred_enum[joint].value

            gt_point = (
                (gt[:, g_idx[0]] + gt[:, g_idx[1]]) / 2
                if isinstance(g_idx, tuple)
                else gt[:, g_idx]
            )
            pred_point = (
                (pred[:, p_idx[0]] + pred[:, p_idx[1]]) / 2
                if isinstance(p_idx, tuple)
                else pred[:, p_idx]
            )

            gt_pts.append(gt_point)
            pred_pts.append(pred_point)

        if not gt_pts or not pred_pts:
            raise ValueError("No valid joints found for evaluation.")

        gt_points = np.stack(gt_pts, axis=1)
        pred_points = np.stack(pred_pts, axis=1)

        distances = np.linalg.norm(gt_points - pred_points, axis=-1)
        correct = distances < (self.threshold * norm_length[:, np.newaxis])

        # --- The key change is here ---
        # Instead of np.mean(correct), we return the per-frame scores.
        # The mean across the joints for each frame is what we want.
        per_frame_pck = np.mean(correct, axis=1) * 100
        return per_frame_pck

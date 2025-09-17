from evaluation.base_calculators import BasePCKCalculator, average_if_tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


class OverallPCKCalculator(BasePCKCalculator):
    def __init__(self, params, gt_enum, pred_enum, verbose=False):
        if "threshold" not in params:
            raise ValueError(
                "Parameter 'threshold' is required for OverallPCKCalculator.")

        threshold = params["threshold"]
        joints_to_evaluate = params.get("joints_to_evaluate", None)

        norm_joints = params.get(
            "norm_joints", self.auto_select_norm_joints(joints_to_evaluate))

        if verbose:
            print(
                f"[OverallPCKCalculator] Using normalization joints: {norm_joints}")

        super().__init__(threshold=threshold, joints_to_evaluate=joints_to_evaluate)

        if gt_enum is None or pred_enum is None:
            raise ValueError("Both gt_enum and pred_enum must be provided.")

        self.gt_enum = gt_enum
        self.pred_enum = pred_enum
        self.norm_joints = norm_joints

    def auto_select_norm_joints(self, joints_to_evaluate):
        if joints_to_evaluate is None:
            return ["LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP", "RIGHT_HIP"]

        joint_set = set(joints_to_evaluate)

        if "LEFT_SHOULDER" in joint_set and "RIGHT_SHOULDER" in joint_set:
            return ["LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP", "RIGHT_HIP"]
        if "LEFT_KNEE" in joint_set and "RIGHT_KNEE" in joint_set:
            return ["LEFT_KNEE", "LEFT_HIP", "RIGHT_KNEE",  "RIGHT_HIP"]
        if "LEFT_HIP" in joint_set and "RIGHT_HIP" in joint_set:
            return ["LEFT_HIP", "RIGHT_HIP"]
        return ["LEFT_HIP", "RIGHT_HIP"]

    def compute(self, gt_keypoints, pred_keypoints):
        gt, pred = np.array(gt_keypoints), np.array(pred_keypoints)
        if pred.shape[-1] == 3:
            pred = pred[..., :2]

        if gt.shape[0] != pred.shape[0] or gt.ndim != 3:
            raise ValueError("Input shape mismatch")

        # Compute normalization lengths
        norm_parts = []
        for i in range(0, len(self.norm_joints), 2):
            try:
                j1 = getattr(self.gt_enum, self.norm_joints[i])
                j2 = getattr(self.gt_enum, self.norm_joints[i + 1])

                p1 = np.array([average_if_tuple(x) for x in gt[:, j1.value]])
                p2 = np.array([average_if_tuple(x) for x in gt[:, j2.value]])
                norm_parts.append(np.linalg.norm(p1 - p2, axis=-1))
            except AttributeError:
                logger.warning(
                    f"Normalization joint missing: {self.norm_joints[i]} or {self.norm_joints[i+1]} — skipping.")
                continue

        if not norm_parts:
            raise ValueError("No valid joint pairs found for normalization.")

        norm_length = np.mean(norm_parts, axis=0)

        if self.joints_to_evaluate is None:
            self.joints_to_evaluate = [j.name for j in self.gt_enum]

        gt_pts, pred_pts = [], []

        for joint in self.joints_to_evaluate:
            if joint not in self.gt_enum.__members__:
                logger.warning(
                    f"Joint '{joint}' not found in GT enum. Skipping.")
                continue
            if joint not in self.pred_enum.__members__:
                logger.warning(
                    f"Joint '{joint}' not found in Pred enum. Skipping.")
                continue

            g_idx, p_idx = self.gt_enum[joint].value, self.pred_enum[joint].value

            if isinstance(g_idx, tuple):
                gt_point = (gt[:, g_idx[0]] + gt[:, g_idx[1]]) / 2
            else:
                gt_point = gt[:, g_idx]

            if isinstance(p_idx, tuple):
                pred_point = (pred[:, p_idx[0]] + pred[:, p_idx[1]]) / 2
            else:
                pred_point = pred[:, p_idx]

            gt_pts.append(gt_point)
            pred_pts.append(pred_point)

        if not gt_pts or not pred_pts:
            raise ValueError("No valid joints found for evaluation.")

        gt_points = np.stack(gt_pts, axis=1)
        pred_points = np.stack(pred_pts, axis=1)

        distances = np.linalg.norm(gt_points - pred_points, axis=-1)
        correct = distances < (self.threshold * norm_length[:, np.newaxis])

        return np.mean(correct) * 100

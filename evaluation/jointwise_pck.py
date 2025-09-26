from evaluation.base_calculators import BasePCKCalculator, average_if_tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


class JointwisePCKCalculator(BasePCKCalculator):
    def __init__(self, params, gt_enum, pred_enum):
        if "threshold" not in params:
            raise ValueError(
                "Parameter 'threshold' is required for JointwisePCKCalculator.")

        threshold = params["threshold"]
        joints_to_evaluate = params.get("joints_to_evaluate", None)
        self.norm_joints = params.get("norm_joints", None)

        super().__init__(threshold=threshold, joints_to_evaluate=joints_to_evaluate)

        if gt_enum is None or pred_enum is None:
            raise ValueError("Both gt_enum and pred_enum must be provided.")

        self.gt_enum = gt_enum
        self.pred_enum = pred_enum

    def _select_norm_joints(self):
        joint_set = set(self.joints_to_evaluate or [])

        if "LEFT_SHOULDER" in joint_set and "RIGHT_SHOULDER" in joint_set:
            return ["LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP", "RIGHT_HIP"]
        elif "LEFT_KNEE" in joint_set and "RIGHT_KNEE" in joint_set:
            return ["LEFT_KNEE", "RIGHT_KNEE", "LEFT_HIP", "RIGHT_HIP"]
        elif "LEFT_HIP" in joint_set and "RIGHT_HIP" in joint_set:
            return ["LEFT_HIP", "RIGHT_HIP"]
        else:
            raise ValueError(
                "Could not determine normalization joints from joints_to_evaluate.")

    def compute(self, gt_keypoints, pred_keypoints):
        gt, pred = np.array(gt_keypoints), np.array(pred_keypoints)
        if pred.shape[1] == 1:
            pred = pred.squeeze(1)
        if self.joints_to_evaluate is None:
            self.joints_to_evaluate = [j.name for j in self.gt_enum]

        # Determine norm_joints
        norm_joints = self.norm_joints or self._select_norm_joints()

        # Compute normalization length
        norm_parts = []
        for i in range(0, len(norm_joints), 2):
            try:
                j1 = getattr(self.gt_enum, norm_joints[i])
                j2 = getattr(self.gt_enum, norm_joints[i + 1])

                p1 = np.array([average_if_tuple(x) for x in gt[:, j1.value]])
                p2 = np.array([average_if_tuple(x) for x in gt[:, j2.value]])
                norm_parts.append(np.linalg.norm(p1 - p2, axis=-1))
            except AttributeError:
                logger.warning(
                    f"Normalization joint missing: {norm_joints[i]} or {norm_joints[i+1]} — skipping.")
                continue

        if not norm_parts:
            raise ValueError("No valid joint pairs found for normalization.")

        norm_length = np.mean(norm_parts, axis=0)

        gt_pts, pred_pts, joint_names = [], [], []

        for joint in self.joints_to_evaluate:
            if joint not in self.gt_enum.__members__:
                logger.warning(
                    f"Joint '{joint}' not found in GT enum. Skipping.")
                continue
            if joint not in self.pred_enum.__members__:
                logger.warning(
                    f"Joint '{joint}' not found in Pred enum. Skipping.")
                continue

            g_idx = self.gt_enum[joint].value
            p_idx = self.pred_enum[joint].value

            gt_point = (gt[:, g_idx[0]] + gt[:, g_idx[1]]) / \
                2 if isinstance(g_idx, tuple) else gt[:, g_idx]
            pred_point = (pred[:, p_idx[0]] + pred[:, p_idx[1]]) / \
                2 if isinstance(p_idx, tuple) else pred[:, p_idx]

            gt_pts.append(gt_point)
            pred_pts.append(pred_point)
            joint_names.append(joint)

        if not joint_names:
            raise ValueError(
                "No valid joints were matched between GT and prediction enums.")

        gt_points = np.stack(gt_pts, axis=1)
        pred_points = np.stack(pred_pts, axis=1)

        # Calculate jointwise distances
        distances = np.linalg.norm(
            gt_points - pred_points, axis=-1) / norm_length[:, np.newaxis]

        # Calculate raw PCK scores for each joint and frame
        jointwise_pck = (distances < self.threshold).astype(int) * 100

        # Aggregate scores by averaging along the frame dimension (axis=0)
        averaged_pck_per_joint = np.mean(jointwise_pck, axis=0)

        # Return the joint names and the new, aggregated scores
        return joint_names, averaged_pck_per_joint

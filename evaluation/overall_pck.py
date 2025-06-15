from evaluation.base_calculators import BasePCKCalculator, average_if_tuple
import numpy as np


class OverallPCKCalculator(BasePCKCalculator):
    def __init__(
        self,
        threshold=0.2,
        joints_to_evaluate=None,
        gt_enum=None,
        pred_enum=None,
        norm_joints=("LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE"),
    ):
        super().__init__(threshold=threshold, joints_to_evaluate=joints_to_evaluate)

        if gt_enum is None or pred_enum is None:
            raise ValueError("Both gt_enum and pred_enum must be provided.")

        self.gt_enum = gt_enum
        self.pred_enum = pred_enum
        self.norm_joints = norm_joints

    def compute(self, gt, pred):
        gt, pred = np.array(gt), np.array(pred)
        if gt.shape[0] != pred.shape[0] or gt.ndim != 3:
            raise ValueError("Input shape mismatch")

        # Normalization length computation using 2 joint pairs
        norm_parts = []
        for i in range(0, len(self.norm_joints), 2):
            j1 = getattr(self.gt_enum, self.norm_joints[i])
            j2 = getattr(self.gt_enum, self.norm_joints[i + 1])

            p1 = np.array([average_if_tuple(x) for x in gt[:, j1.value]])
            p2 = np.array([average_if_tuple(x) for x in gt[:, j2.value]])
            norm_parts.append(np.linalg.norm(p1 - p2, axis=-1))

        norm_length = np.mean(norm_parts, axis=0)  # Shape: (N,)

        # Evaluate all specified joints
        if self.joints_to_evaluate is None:
            self.joints_to_evaluate = [j.name for j in self.gt_enum]

        gt_pts, pred_pts = [], []
        for joint in self.joints_to_evaluate:
            if (
                joint in self.pred_enum.__members__
                and joint in self.gt_enum.__members__
            ):
                g_idx, p_idx = self.gt_enum[joint].value, self.pred_enum[joint].value
                gt_pts.append(
                    (gt[:, g_idx[0]] + gt[:, g_idx[1]]) / 2
                    if isinstance(g_idx, tuple)
                    else gt[:, g_idx]
                )
                pred_pts.append(
                    (pred[:, p_idx[0]] + pred[:, p_idx[1]]) / 2
                    if isinstance(p_idx, tuple)
                    else pred[:, p_idx]
                )

        gt_points = np.stack(gt_pts, axis=1)
        pred_points = np.stack(pred_pts, axis=1)

        distances = np.linalg.norm(gt_points - pred_points, axis=-1)
        correct = distances < (self.threshold * norm_length[:, np.newaxis])

        return np.mean(correct) * 100

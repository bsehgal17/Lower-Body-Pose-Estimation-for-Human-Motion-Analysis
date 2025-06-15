from evaluation.base_calculators import BasePCKCalculator
import numpy as np


class JointwisePCKCalculator(BasePCKCalculator):
    def __init__(
        self, threshold=0.2, joints_to_evaluate=None, gt_enum=None, pred_enum=None
    ):
        super().__init__(threshold=threshold, joints_to_evaluate=joints_to_evaluate)
        if gt_enum is None or pred_enum is None:
            raise ValueError("Both gt_enum and pred_enum must be provided.")

        self.gt_enum = gt_enum
        self.pred_enum = pred_enum

    def compute(self, gt, pred):
        gt, pred = np.array(gt), np.array(pred)

        # Default to all joint names from GT enum if not specified
        if self.joints_to_evaluate is None:
            self.joints_to_evaluate = [j.name for j in self.gt_enum]

        # Default normalization: distance between first two joints
        norm_joint_1 = getattr(self.gt_enum, self.joints_to_evaluate[0])
        norm_joint_2 = getattr(self.gt_enum, self.joints_to_evaluate[1])
        norm_idx1 = norm_joint_1.value
        norm_idx2 = norm_joint_2.value
        norm_ref_1 = (
            (gt[:, norm_idx1[0]] + gt[:, norm_idx1[1]]) / 2
            if isinstance(norm_idx1, tuple)
            else gt[:, norm_idx1]
        )
        norm_ref_2 = (
            (gt[:, norm_idx2[0]] + gt[:, norm_idx2[1]]) / 2
            if isinstance(norm_idx2, tuple)
            else gt[:, norm_idx2]
        )
        norm_length = np.linalg.norm(norm_ref_1 - norm_ref_2, axis=-1)

        gt_pts, pred_pts, joint_names = [], [], []

        for joint in self.joints_to_evaluate:
            if (
                joint in self.pred_enum.__members__
                and joint in self.gt_enum.__members__
            ):
                g_idx = self.gt_enum[joint].value
                p_idx = self.pred_enum[joint].value

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
                joint_names.append(joint)

        gt_points = np.stack(gt_pts, axis=1)
        pred_points = np.stack(pred_pts, axis=1)

        distances = (
            np.linalg.norm(gt_points - pred_points, axis=-1)
            / norm_length[:, np.newaxis]
        )
        jointwise_pck = (distances < self.threshold).astype(int) * 100
        return joint_names, jointwise_pck

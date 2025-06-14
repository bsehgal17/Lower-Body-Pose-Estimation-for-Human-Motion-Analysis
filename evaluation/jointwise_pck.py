from evaluation.base_calculators import BasePCKCalculator
from utils.joint_enum import GTJoints, PredJoints
import numpy as np


class JointwisePCKCalculator(BasePCKCalculator):
    def compute(self, gt, pred):
        gt, pred = np.array(gt), np.array(pred)
        if self.joints_to_evaluate is None:
            self.joints_to_evaluate = [j.name for j in GTJoints]
            norm_length = np.linalg.norm(
                gt[:, GTJoints.LEFT_SHOULDER.value] - gt[:, GTJoints.RIGHT_HIP.value],
                axis=-1,
            )
        else:
            norm_length = np.linalg.norm(
                gt[:, GTJoints.RIGHT_HIP.value] - gt[:, GTJoints.LEFT_HIP.value],
                axis=-1,
            )

        gt_pts, pred_pts, joint_names = [], [], []
        for joint in self.joints_to_evaluate:
            if joint in PredJoints.__members__:
                g_idx, p_idx = GTJoints[joint].value, PredJoints[joint].value
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
                joint_names.append(joint)

        gt_points = np.stack(gt_pts, axis=1)
        pred_points = np.stack(pred_pts, axis=1)

        distances = (
            np.linalg.norm(gt_points - pred_points, axis=-1)
            / norm_length[:, np.newaxis]
        )
        jointwise_pck = (distances < self.threshold).astype(int) * 100
        return joint_names, jointwise_pck

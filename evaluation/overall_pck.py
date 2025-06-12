from evaluation.base_calculators import BasePCKCalculator, average_if_tuple
from utils.joint_enum import GTJoints, PredJoints
import numpy as np


class OverallPCKCalculator(BasePCKCalculator):
    def compute(self, gt, pred):
        gt, pred = np.array(gt), np.array(pred)
        if gt.shape[0] != pred.shape[0] or gt.ndim != 3:
            raise ValueError("Input shape mismatch")

        if self.joints_to_evaluate is None:
            self.joints_to_evaluate = [j.name for j in GTJoints]
            left_shoulder = np.array(
                [average_if_tuple(x) for x in gt[:, GTJoints.LEFT_SHOULDER.value]]
            )
            right_shoulder = np.array(
                [average_if_tuple(x) for x in gt[:, GTJoints.RIGHT_SHOULDER.value]]
            )
            left_hip = np.array(
                [average_if_tuple(x) for x in gt[:, GTJoints.LEFT_HIP.value]]
            )
            right_hip = np.array(
                [average_if_tuple(x) for x in gt[:, GTJoints.RIGHT_HIP.value]]
            )
            norm_length = (
                np.linalg.norm(left_shoulder - left_hip, axis=-1)
                + np.linalg.norm(right_shoulder - right_hip, axis=-1)
            ) / 2
        else:
            left_hip = np.array(
                [average_if_tuple(x) for x in gt[:, GTJoints.LEFT_HIP.value]]
            )
            right_hip = np.array(
                [average_if_tuple(x) for x in gt[:, GTJoints.RIGHT_HIP.value]]
            )
            left_knee = np.array(
                [average_if_tuple(x) for x in gt[:, GTJoints.LEFT_KNEE.value]]
            )
            right_knee = np.array(
                [average_if_tuple(x) for x in gt[:, GTJoints.RIGHT_KNEE.value]]
            )
            norm_length = (
                np.linalg.norm(left_hip - left_knee, axis=-1)
                + np.linalg.norm(right_hip - right_knee, axis=-1)
            ) / 2

        gt_pts, pred_pts = [], []
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

        gt_points = np.stack(gt_pts, axis=1)
        pred_points = np.stack(pred_pts, axis=1)

        distances = np.linalg.norm(gt_points - pred_points, axis=-1)
        correct = distances < (self.threshold * norm_length[:, np.newaxis])

        return np.mean(correct) * 100

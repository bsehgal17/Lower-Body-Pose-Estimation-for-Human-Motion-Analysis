import numpy as np
import logging
from enum import Enum
# NOTE: This is a placeholder for your actual BaseCalculator class

logger = logging.getLogger(__name__)


class BaseCalculator:
    def __init__(self, **kwargs):
        pass


def _get_joint_point(keypoints_list, index):
    """
    Helper function to get a single joint point from a list of keypoints.
    If the index is a tuple, it averages the points at those indices.
    """
    if isinstance(index, (list, tuple)):
        point1 = np.array(keypoints_list[index[0]])
        point2 = np.array(keypoints_list[index[1]])
        return (point1 + point2) / 2
    else:
        return np.array(keypoints_list[index])


class JointwiseAPCalculator(BaseCalculator):
    def __init__(self, params, gt_enum, pred_enum, verbose=False):
        """
        Initializes the Joint-wise AP calculator.

        Args:
            params (dict): A dictionary of parameters, including 'kpt_sigmas'
                           and 'joints_to_evaluate'.
            gt_enum (Enum): An enum mapping ground truth joint names to indices.
            pred_enum (Enum): An enum mapping predicted joint names to indices.
            verbose (bool): If True, prints additional information.
        """
        super().__init__(**params)

        if "kpt_sigmas" not in params:
            raise ValueError(
                "Parameter 'kpt_sigmas' is required for JointwiseAPCalculator.")
        if "joints_to_evaluate" not in params:
            raise ValueError(
                "Parameter 'joints_to_evaluate' is required for JointwiseAPCalculator.")

        # Get the joints to evaluate directly from the parameters and convert to lowercase for consistency.
        self.joints_to_evaluate = []
        for item in params["joints_to_evaluate"]:
            if isinstance(item, (list, tuple)):
                self.joints_to_evaluate.extend([j.lower() for j in item])
            else:
                self.joints_to_evaluate.append(item.lower())

        self.kpt_sigmas = {
            k.lower(): v for k, v in params["kpt_sigmas"].items()}
        self.gt_enum = gt_enum
        self.pred_enum = pred_enum
        self.verbose = verbose

        # Use a single OKS threshold for determining "correctness" of a joint prediction
        # for a given GT pose. This is analogous to the PCK threshold.
        self.oks_threshold = params.get('oks_threshold', 0.5)

    def _calculate_oks_for_joint(self, gt_kpt, pred_kpt, s, sigma):
        """
        Calculates the Object Keypoint Similarity (OKS) for a single joint.

        Args:
            gt_kpt (np.array): Ground truth keypoint (2D).
            pred_kpt (np.array): Predicted keypoint (2D).
            s (float): The scale of the person (sqrt of bounding box area).
            sigma (float): The normalization constant for the joint.

        Returns:
            float: The OKS score for the single joint.
        """
        if s == 0 or sigma == 0:
            return 0.0

        distance_sq = np.sum((gt_kpt - pred_kpt)**2)
        return np.exp(-distance_sq / (2 * s**2 * sigma**2))

    def compute(self, gt_keypoints, gt_bboxes, gt_scores, pred_keypoints, pred_bboxes, pred_scores):
        """
        Computes the AP for each joint across all predictions.

        Args:
            gt_keypoints (list): List of ground truth keypoints for each pose.
            gt_bboxes (list): List of ground truth bounding boxes.
            pred_keypoints (list): List of predicted keypoints for each pose.
            pred_bboxes (list): List of predicted bounding boxes.
            pred_scores (list): List of predicted pose scores.

        Returns:
            dict: A dictionary of AP results for each joint.
        """
        # Create a combined list of all ground truth and prediction instances
        gt_poses = []
        for i in range(len(gt_keypoints)):
            if gt_keypoints[i] is not None:
                bbox_area = (gt_bboxes[i][2] - gt_bboxes[i][0]) * (gt_bboxes[i][3] - gt_bboxes[i][1]
                                                                   ) if gt_bboxes is not None and i < len(gt_bboxes) and gt_bboxes[i] is not None else 1.0
                gt_poses.append({
                    'keypoints': np.array(gt_keypoints[i]),
                    'bbox_area': bbox_area,
                })

        pred_poses = []
        for i in range(len(pred_keypoints)):
            if pred_keypoints[i] is not None:
                bbox_area = (pred_bboxes[i][0][2] - pred_bboxes[i][0][0]) * (pred_bboxes[i][0][3] - pred_bboxes[i]
                                                                             [0][1]) if pred_bboxes is not None and i < len(pred_bboxes) and pred_bboxes[i] is not None else 1.0
                pred_poses.append({
                    'keypoints': np.array(pred_keypoints[i]),
                    'bbox_area': bbox_area,
                    'score': pred_scores[i]
                })

        # Ensure we have common joints to evaluate
        gt_joints_set = set(j.lower() for j in self.gt_enum.__members__)
        pred_joints_set = set(j.lower() for j in self.pred_enum.__members__)
        common_joints = [
            joint for joint in self.joints_to_evaluate
            if joint in gt_joints_set and joint in pred_joints_set and joint in self.kpt_sigmas
        ]

        if not gt_poses or not pred_poses or not common_joints:
            logger.warning(
                "No ground truth, predictions, or common joints to evaluate.")
            return {joint: 0.0 for joint in common_joints}

        jointwise_ap_results = {}

        # Iterate over each joint to calculate its individual AP
        for joint_name in common_joints:
            pred_list = []

            # For each prediction, find the best-matching GT for this *specific joint*
            for pred_pose in pred_poses:
                best_oks = 0.0

                pred_kpt_idx = self.pred_enum[joint_name.upper()].value
                pred_kpt = _get_joint_point(
                    pred_pose['keypoints'], pred_kpt_idx)
                pred_s = np.sqrt(pred_pose['bbox_area'])

                # We need a way to track which GT joints have been matched to a pred joint
                # to avoid double counting, but this is complicated. A simpler approach
                # is to iterate through all GT poses and find the best match.
                for gt_pose in gt_poses:
                    gt_kpt_idx = self.gt_enum[joint_name.upper()].value
                    gt_kpt = _get_joint_point(gt_pose['keypoints'], gt_kpt_idx)
                    gt_s = np.sqrt(gt_pose['bbox_area'])

                    sigma = self.kpt_sigmas[joint_name]
                    current_oks = self._calculate_oks_for_joint(
                        gt_kpt, pred_kpt, gt_s, sigma)
                    if current_oks > best_oks:
                        best_oks = current_oks

                # Now, determine if this prediction is a 'true positive' for the current joint
                # by checking if its best OKS score exceeds the threshold.
                is_correct = 1 if best_oks >= self.oks_threshold else 0

                pred_list.append({
                    'score': pred_pose['score'],
                    'is_correct': is_correct
                })

            # Sort predictions for this joint by score
            pred_list.sort(key=lambda x: x['score'], reverse=True)

            true_positives = np.zeros(len(pred_list))
            false_positives = np.zeros(len(pred_list))

            # Fill TP/FP arrays
            for i, pred_item in enumerate(pred_list):
                if pred_item['is_correct'] == 1:
                    true_positives[i] = 1
                else:
                    false_positives[i] = 1

            tp_cumsum = np.cumsum(true_positives)
            fp_cumsum = np.cumsum(false_positives)

            # Calculate recall and precision
            num_gt_joints = len(gt_poses)
            recall = tp_cumsum / (num_gt_joints + 1e-10)
            precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)

            # Calculate AP using numerical integration (trapezoidal rule)
            ap = np.trapz(precision, recall)
            jointwise_ap_results[joint_name] = ap * 100

        return jointwise_ap_results

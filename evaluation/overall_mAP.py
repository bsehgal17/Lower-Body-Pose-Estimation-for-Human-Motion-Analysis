import numpy as np
import logging
from enum import Enum

logger = logging.getLogger(__name__)

# NOTE: This is a placeholder for your actual BasePCKCalculator class
# that would contain shared logic like initialization.


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


class MAPCalculator(BaseCalculator):
    def __init__(self, params, gt_enum, pred_enum, verbose=False):
        """
        Initializes the mAP calculator for lower body pose estimation.

        Args:
            params (dict): A dictionary of parameters. Must include 'kpt_sigmas'
                           and 'joints_to_evaluate'.
            gt_enum (Enum): An enum mapping ground truth joint names to indices.
            pred_enum (Enum): An enum mapping predicted joint names to indices.
            verbose (bool): If True, prints additional information.
        """
        # Call the base class initializer
        super().__init__(**params)

        # Check for required parameters. This enforces a complete YAML config.
        if "kpt_sigmas" not in params:
            raise ValueError(
                "Parameter 'kpt_sigmas' is required for MAPCalculator.")
        if "joints_to_evaluate" not in params:
            raise ValueError(
                "Parameter 'joints_to_evaluate' is required for MAPCalculator.")

        # Get the joints to evaluate directly from the parameters.
        # This section now handles single joint names as well as tuples of joints.
        # It flattens the list for easier processing later.
        self.joints_to_evaluate = []
        for item in params["joints_to_evaluate"]:
            if isinstance(item, (list, tuple)):
                self.joints_to_evaluate.extend(item)
            else:
                self.joints_to_evaluate.append(item)

        # Use the provided keypoint sigmas for the OKS calculation.
        # These are crucial constants that normalize distance by joint difficulty.
        self.kpt_sigmas = {
            k.lower(): v for k, v in params["kpt_sigmas"].items()}

        self.gt_enum = gt_enum
        self.pred_enum = pred_enum
        self.verbose = verbose

    def _calculate_oks(self, gt, pred, s, kpt_sigmas, common_joints):
        """
        Calculates the Object Keypoint Similarity (OKS) for a single instance.

        Args:
            gt (np.array): Ground truth keypoints (N x 2).
            pred (np.array): Predicted keypoints (N x 2).
            s (float): The scale of the person (sqrt of bounding box area).
            kpt_sigmas (dict): Per-keypoint normalization constants.
            common_joints (list): The list of common joint names to evaluate.

        Returns:
            float: The OKS score.
        """
        # If scale is zero, OKS is undefined, return 0.
        if s == 0:
            return 0.0

        # Ensure keypoints are 2D arrays (N, 2)
        gt = np.atleast_2d(gt)
        pred = np.atleast_2d(pred)

        # Calculate squared Euclidean distances for all common keypoints
        distances_sq = np.sum((gt - pred)**2, axis=1)

        # Apply the OKS formula
        numerator = 0
        denominator = 0

        # Iterate over the joints we care about for OKS calculation, using the common joints list
        for i, joint_name in enumerate(common_joints):
            # Check if the joint exists and is valid
            if joint_name in kpt_sigmas:
                # Get the sigma value for the current joint
                sigma = kpt_sigmas[joint_name]
                # The exponent term is di^2 / (2 * s**2 * ki**2)
                numerator += np.exp(-distances_sq[i] / (2 * s**2 * sigma**2))
                denominator += 1

        return numerator / denominator if denominator > 0 else 0.0

    def compute(self, gt_poses, pred_poses):
        """
        Computes the overall mAP for lower body pose estimation across a video.

        Args:
            gt_poses (list): List of dictionaries containing ground truth poses.
                             Each dict should have 'keypoints' and 'bbox_area'.
            pred_poses (list): List of dictionaries containing predicted poses.
                               Each dict should have 'keypoints', 'bbox_area',
                               and 'score' (confidence).

        Returns:
            dict: A dictionary of mAP results.
        """
        # Ensure gt_poses is in the correct format (list of dicts)
        if isinstance(gt_poses, np.ndarray):
            gt_poses_list = []
            for pose_kpts in gt_poses:
                gt_poses_list.append({
                    'keypoints': pose_kpts,
                    'bbox_area': 1.0,  # Placeholder
                    'matched': False
                })
            gt_poses = gt_poses_list

        # Ensure pred_poses is in the correct format (list of dicts)
        if isinstance(pred_poses, np.ndarray):
            # Convert the numpy array to a list of dictionaries
            # Note: We're using placeholder values for score and bbox_area
            # as they are not present in the numpy array.
            pred_poses_list = []
            for pose_kpts in pred_poses:
                pred_poses_list.append({
                    'keypoints': pose_kpts,
                    'bbox_area': 1.0,  # Placeholder
                    'score': 1.0
                })
            pred_poses = pred_poses_list

        # Store results for each OKS threshold
        ap_results = {}
        oks_thresholds = np.arange(0.5, 1.0, 0.05)

        # Determine the set of common joints to evaluate
        gt_joints_set = set(self.gt_enum.__members__)
        pred_joints_set = set(self.pred_enum.__members__)

        common_joints = [
            joint for joint in self.joints_to_evaluate
            if joint in gt_joints_set and joint in pred_joints_set
        ]

        if not gt_poses or not pred_poses or not common_joints:
            logger.warning(
                "No ground truth, predictions, or common joints to evaluate.")
            return {"mAP": 0.0, "mAP_50": 0.0, "mAP_75": 0.0, "ap_per_threshold": ap_results, "individual_oks_scores": []}

        # Match predictions to ground truth
        oks_scores = []
        for pred_pose in pred_poses:
            best_oks = 0.0
            best_gt_idx = -1

            # This is the updated section to handle single or tuple indices for a joint.
            pred_kpts = np.array([
                _get_joint_point(pred_pose.get(
                    'keypoints', []), self.pred_enum[j].value)
                for j in common_joints
            ])
            pred_score = pred_pose.get('score', 0)
            pred_s = np.sqrt(pred_pose.get('bbox_area', 0))

            # Find the best matching ground truth pose
            for i, gt_pose in enumerate(gt_poses):
                # Ensure the ground truth pose hasn't been matched yet
                if not gt_pose.get('matched', False):
                    # This is the updated section to handle single or tuple indices for a joint.
                    gt_kpts = np.array([
                        _get_joint_point(gt_pose.get(
                            'keypoints', []), self.gt_enum[j].value)
                        for j in common_joints
                    ])
                    gt_s = np.sqrt(gt_pose.get('bbox_area', 0))

                    current_oks = self._calculate_oks(
                        gt_kpts, pred_kpts, gt_s, self.kpt_sigmas, common_joints)
                    if current_oks > best_oks:
                        best_oks = current_oks
                        best_gt_idx = i

            # If a match is found, store the result
            if best_oks > 0 and best_gt_idx != -1:
                oks_scores.append(
                    {'score': pred_score, 'oks': best_oks, 'is_match': True})
                gt_poses[best_gt_idx]['matched'] = True
            else:
                oks_scores.append(
                    {'score': pred_score, 'oks': 0.0, 'is_match': False})

        # Sort predictions by confidence score
        oks_scores.sort(key=lambda x: x['score'], reverse=True)

        # Calculate AP for each threshold
        for t in oks_thresholds:
            true_positives = np.zeros(len(oks_scores))
            false_positives = np.zeros(len(oks_scores))

            for i, score in enumerate(oks_scores):
                if score['oks'] >= t:
                    true_positives[i] = 1
                else:
                    false_positives[i] = 1

            # Cumulative sum
            tp_cumsum = np.cumsum(true_positives)
            fp_cumsum = np.cumsum(false_positives)

            recall = tp_cumsum / \
                len(gt_poses) if len(gt_poses) > 0 else np.zeros_like(tp_cumsum)
            # Add epsilon to avoid division by zero
            precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)

            # Compute AP using the 11-point interpolation method (or simpler)
            ap = np.trapz(precision, recall)
            ap_results[f"AP@{t:.2f}"] = ap

        # Calculate mAP and mAP_50 and mAP_75
        mAP = np.mean(list(ap_results.values()))
        mAP_50 = ap_results.get("AP@0.50", 0)
        mAP_75 = ap_results.get("AP@0.75", 0)

        return {
            "mAP": mAP * 100,
            "mAP_50": mAP_50 * 100,
            "mAP_75": mAP_75 * 100,
            "ap_per_threshold": ap_results,
            "individual_oks_scores": oks_scores,
        }

import numpy as np
import logging
from enum import Enum

logger = logging.getLogger(__name__)

# NOTE: This is a placeholder for your actual BasePCKCalculator class
# that would contain shared logic like initialization.


class BaseCalculator:
    def __init__(self, **kwargs):
        pass


class MAPCalculator(BaseCalculator):
    def __init__(self, params, gt_enum, pred_enum, verbose=False):
        """
        Initializes the mAP calculator for lower body pose estimation.

        Args:
            params (dict): A dictionary of parameters. Must include 'kpt_sigmas'.
            gt_enum (Enum): An enum mapping ground truth joint names to indices.
            pred_enum (Enum): An enum mapping predicted joint names to indices.
            verbose (bool): If True, prints additional information.
        """
        # Call the base class initializer
        super().__init__(**params)

        if "kpt_sigmas" not in params:
            raise ValueError(
                "Parameter 'kpt_sigmas' is required for MAPCalculator.")

        # Define the lower body keypoints to evaluate.
        # These correspond to the standard COCO keypoints.
        self.joints_to_evaluate = [
            "left_hip", "right_hip",
            "left_knee", "right_knee",
            "left_ankle", "right_ankle"
        ]

        # Use the provided keypoint sigmas for the OKS calculation.
        # These are crucial constants that normalize distance by joint difficulty.
        self.kpt_sigmas = {
            k.lower(): v for k, v in params["kpt_sigmas"].items()}

        self.gt_enum = gt_enum
        self.pred_enum = pred_enum
        self.verbose = verbose

    def _calculate_oks(self, gt, pred, s, kpt_sigmas):
        """
        Calculates the Object Keypoint Similarity (OKS) for a single instance.

        Args:
            gt (np.array): Ground truth keypoints (N x 2).
            pred (np.array): Predicted keypoints (N x 2).
            s (float): The scale of the person (sqrt of bounding box area).
            kpt_sigmas (dict): Per-keypoint normalization constants.

        Returns:
            float: The OKS score.
        """
        # If scale is zero, OKS is undefined, return 0.
        if s == 0:
            return 0.0

        # Ensure keypoints are 2D arrays (N, 2)
        gt = np.atleast_2d(gt)
        pred = np.atleast_2d(pred)

        # Calculate squared Euclidean distances for all keypoints
        distances_sq = np.sum((gt - pred)**2, axis=1)

        # Apply the OKS formula, only for the lower body keypoints
        numerator = 0
        denominator = 0

        # Iterate over the joints we care about for OKS calculation
        for i, joint_name in enumerate(self.joints_to_evaluate):
            # Check if the joint exists and is valid
            if joint_name in kpt_sigmas:
                # Get the sigma value for the current joint
                sigma = kpt_sigmas[joint_name]
                # The k_i value in the formula is sigma.
                # The exponent term is di^2 / (2 * s^2 * ki^2)
                numerator += np.exp(-distances_sq[i] / (2 * s**2 * sigma**2))
                denominator += 1  # We're only evaluating a fixed set of joints

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
        # Store results for each OKS threshold
        ap_results = {}
        oks_thresholds = np.arange(0.5, 1.0, 0.05)

        if not gt_poses or not pred_poses:
            logger.warning("No ground truth or predictions to evaluate.")
            return {"mAP": 0.0, "mAP_50": 0.0, "mAP_75": 0.0, "ap_per_threshold": ap_results, "individual_oks_scores": []}

        # Match predictions to ground truth
        oks_scores = []
        for pred_pose in pred_poses:
            best_oks = 0.0
            best_gt_idx = -1

            # Use .get() to avoid key errors and provide a default
            pred_kpts = np.array(pred_pose.get('keypoints', []))
            pred_score = pred_pose.get('score', 0)
            pred_s = np.sqrt(pred_pose.get('bbox_area', 0))

            # Find the best matching ground truth pose
            for i, gt_pose in enumerate(gt_poses):
                gt_kpts = np.array(gt_pose.get('keypoints', []))
                gt_s = np.sqrt(gt_pose.get('bbox_area', 0))

                # Make sure the ground truth pose hasn't been matched yet
                if 'matched' not in gt_pose:
                    current_oks = self._calculate_oks(
                        gt_kpts, pred_kpts, gt_s, self.kpt_sigmas)
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

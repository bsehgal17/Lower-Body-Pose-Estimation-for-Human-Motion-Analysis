import numpy as np
from utils.joint_enum import GTJoints, PredJoints


def average_if_tuple(value):
    """Check if value is shaped (N, 2, 2) and return the average if true."""
    value = np.array(value)
    if value.shape == (2, 2):  # Ensures correct averaging for 2D points (x, y)
        return np.mean(value, axis=0)
    return value


def compute_pck(gt, pred, threshold=0.05, joints_to_evaluate=None):
    """
    Compute Percentage of Correct Keypoints (PCK) for pose estimation.

    Parameters:
    - gt: Ground truth keypoints (NxJx2) or (NxJx3) array
    - pred: Predicted keypoints (NxJx2) or (NxJx3) array
    - threshold: Scaling factor for correctness (default: 5% of normalization length)
    - joints_to_evaluate: List of joint names to evaluate (lower body if provided, full body otherwise)

    Returns:
    - PCK value (percentage)
    """
    # Convert to numpy and check shape
    gt, pred = np.array(gt, dtype=np.float64), np.array(pred, dtype=np.float64)
    if gt.shape[0] != pred.shape[0] or gt.ndim != 3:
        raise ValueError(
            "Shapes of gt and pred must be (N, J, 2) or (N, J, 3)")

    if joints_to_evaluate is None:
        # Full-body evaluation
        joints_to_evaluate = [joint.name for joint in GTJoints]
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
        # left_knee = np.array(
        #     [average_if_tuple(x) for x in gt[:, GTJoints.LEFT_ANKLE.value]]
        # )
        # right_knee = np.array(
        #     [average_if_tuple(x) for x in gt[:, GTJoints.RIGHT_ANKLE.value]]
        # )
        norm_length = (
            np.linalg.norm(left_hip - left_knee, axis=-1)
            + np.linalg.norm(right_hip - right_knee, axis=-1)
        ) / 2

    pred_indices = []
    gt_indices = []

    for joint in joints_to_evaluate:
        if joint in PredJoints.__members__:  # Check if joint exists in PredJoints
            gt_joint = GTJoints[joint].value
            pred_joint = PredJoints[joint].value

            if isinstance(gt_joint, tuple):  # If joint has two points, compute midpoint
                gt_indices.append(
                    (gt[:, gt_joint[0]] + gt[:, gt_joint[1]]) / 2)
            else:
                gt_indices.append(gt[:, gt_joint])

            if isinstance(
                pred_joint, tuple
            ):  # If pred joint has two points, compute midpoint
                pred_indices.append(
                    (pred[:, pred_joint[0]] + pred[:, pred_joint[1]]) / 2
                )
            else:
                pred_indices.append(pred[:, pred_joint])

    if not gt_indices or not pred_indices:
        raise ValueError("No valid joints found for evaluation.")

    gt_points = np.stack(gt_indices, axis=1)  # Ensure shape consistency
    pred_points = np.stack(pred_indices, axis=1)

    # Compute distances and correctness
    distances = np.linalg.norm(gt_points - pred_points, axis=-1)
    correct = distances < (threshold * norm_length[:, np.newaxis])

    # Compute final PCK
    return np.mean(correct) * 100

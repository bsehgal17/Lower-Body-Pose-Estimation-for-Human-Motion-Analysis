import numpy as np
from joint_enum import GTJoints, PredJoints


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
    if gt.shape != pred.shape or gt.ndim != 3:
        raise ValueError("Shapes of gt and pred must be (N, J, 2) or (N, J, 3)")

    if joints_to_evaluate is None:
        # Full-body evaluation
        joints_to_evaluate = [joint.name for joint in GTJoints]
        pelvis = gt[:, GTJoints.PELVIS.value]
        head = (
            gt[:, GTJoints.HEAD.value[0]] + gt[:, GTJoints.HEAD.value[1]]
        ) / 2  # Head midpoint
        norm_length = np.linalg.norm(pelvis - head, axis=-1)  # Pelvis to head
    else:
        # Lower-body evaluation
        pelvis = gt[:, GTJoints.PELVIS.value]
        ankle = gt[:, GTJoints.RIGHT_ANKLE.value]
        norm_length = np.linalg.norm(pelvis - ankle, axis=-1)  # Pelvis to ankle

    gt_indices = []
    pred_indices = []
    gt_points = []

    for joint in joints_to_evaluate:
        if joint in PredJoints.__members__:  # Check if joint exists in PredJoints
            gt_joint = GTJoints[joint].value
            pred_joint = PredJoints[joint].value

            if isinstance(
                gt_joint[0], tuple
            ):  # If joint has two points, compute midpoint
                gt_points.append((gt[:, gt_joint[0][0]] + gt[:, gt_joint[1][0]]) / 2)
            else:
                gt_points.append(gt[:, gt_joint])

            if isinstance(
                pred_joint, tuple
            ):  # If pred joint has two points, compute midpoint
                pred_indices.append(
                    (pred[:, pred_joint[0]] + pred[:, pred_joint[1]]) / 2
                )
            else:
                pred_indices.append(pred[:, pred_joint])

    if not gt_points or not pred_indices:
        raise ValueError("No valid joints found for evaluation.")

    gt_points = np.stack(gt_points, axis=1)  # Ensure shape consistency
    pred_points = np.stack(pred_indices, axis=1)

    # Compute distances and correctness
    distances = np.linalg.norm(gt_points - pred_points, axis=-1)
    correct = distances < threshold * norm_length[:, np.newaxis]

    # Compute final PCK
    return np.mean(correct) * 100

import numpy as np


def select_norm_joints(joints_to_evaluate):
    """
    Select normalization joints based on joints_to_evaluate.
    Args:
        joints_to_evaluate: list of joint names
    Returns:
        norm_joints: list of joint names (pairs)
    """
    if joints_to_evaluate is None:
        return ["LEFT_SHOULDER", "RIGHT_HIP", "RIGHT_SHOULDER", "LEFT_HIP"]
    joint_set = set(joints_to_evaluate)
    if "LEFT_SHOULDER" in joint_set and "RIGHT_SHOULDER" in joint_set:
        return ["LEFT_SHOULDER", "RIGHT_HIP", "RIGHT_SHOULDER", "LEFT_HIP"]
    if "LEFT_KNEE" in joint_set and "RIGHT_KNEE" in joint_set:
        return ["LEFT_KNEE", "LEFT_HIP", "RIGHT_KNEE", "RIGHT_HIP"]
    if "LEFT_HIP" in joint_set and "RIGHT_HIP" in joint_set:
        return ["LEFT_HIP", "RIGHT_HIP"]
    raise ValueError(
        "Could not determine normalization joints from joints_to_evaluate."
    )


def compute_norm_length(gt_enum, norm_joints, gt_keypoints):
    """
    Compute normalization length for PCK calculation.
    Args:
        gt_enum: Enum or object with joint indices as attributes.
        norm_joints: List of joint names (pairs).
        gt_keypoints: np.ndarray of shape (..., num_joints, 2 or 3)
    Returns:
        norm_length: np.ndarray of normalization lengths
    """
    norm_parts = []
    for i in range(0, len(norm_joints), 2):
        try:
            j1 = getattr(gt_enum, norm_joints[i])
            j2 = getattr(gt_enum, norm_joints[i + 1])
            p1 = gt_keypoints[..., j1, :]
            p2 = gt_keypoints[..., j2, :]
            norm_parts.append(np.linalg.norm(p1 - p2, axis=-1))
        except Exception:
            print(
                f"Normalization joint missing: {norm_joints[i]} or {norm_joints[i + 1]} — skipping."
            )
    if not norm_parts:
        raise ValueError("No valid joint pairs found for normalization.")
    norm_length = np.mean(norm_parts, axis=0)
    return norm_length

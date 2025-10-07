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
        "Could not determine normalization joints from joints_to_evaluate.")


def compute_norm_length(gt_enum, norm_joints, gt_keypoints):
    """
    Compute normalization length for PCK calculation.
    Handles enums where joint indices can be single ints or tuples.
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

            # --- Handle both single index and tuple indices ---
            def get_joint_avg(j):
                if isinstance(j.value, tuple):
                    idxs = np.array(j.value)
                    points = gt_keypoints[..., idxs, :]
                    return np.nanmean(points, axis=-2)  # average both joints
                else:
                    return gt_keypoints[..., j.value, :]

            p1 = get_joint_avg(j1)
            p2 = get_joint_avg(j2)

            norm_parts.append(np.linalg.norm(p1 - p2, axis=-1))

        except Exception as e:
            print(f"⚠️ Normalization joint missing or invalid: "
                  f"{norm_joints[i]} or {norm_joints[i + 1]} — skipping. ({e})")

    if not norm_parts:
        raise ValueError("No valid joint pairs found for normalization.")

    norm_length = np.nanmean(norm_parts, axis=0)
    return norm_length

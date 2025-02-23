import numpy as np


def compute_pck(gt, pred, threshold=0.05):
    """
    Compute the Percentage of Correct Keypoints (PCK) for lower-body pose estimation,
    considering hip, knee, ankle, heel, and toe joints.

    Parameters:
    - gt: Ground truth keypoints (NxJx2) or (NxJx3) array (J = number of joints)
    - pred: Predicted keypoints (NxJx2) or (NxJx3) array
    - threshold: Scaling factor for correctness (default: 5% of lower body length)

    Returns:
    - PCK value (percentage)
    """
    # Define lower-body joint indices
    hip_idx, knee_idx, ankle_idx = 0, 1, 2

    # Compute lower body length (thigh + shin + foot including heel & toes)
    thigh_length = np.linalg.norm(gt[:, knee_idx] - gt[:, hip_idx], axis=-1)
    shin_length = np.linalg.norm(gt[:, ankle_idx] - gt[:, knee_idx], axis=-1)

    lower_body_length = thigh_length + shin_length

    # Compute correctness based on lower body length threshold
    correct = np.linalg.norm(
        gt - pred, axis=-1) < threshold * lower_body_length[:, None]

    return np.mean(correct) * 100  # Convert to percentage


def compute_mpjpe(gt: np.ndarray, pred: np.ndarray) -> float:
    """
    Compute the Mean Per Joint Position Error (MPJPE) for lower-body pose estimation.

    Parameters:
    - gt (np.ndarray): Ground truth joint positions (NxJx2) for 2D or (NxJx3) for 3D.
    - pred (np.ndarray): Predicted joint positions (NxJx2) or (NxJx3).

    Returns:
    - float: MPJPE value (mean Euclidean distance between predicted and ground truth joints).
    """
    print("Computing MPJPE for lower body...")
    return np.mean(np.linalg.norm(gt - pred, axis=-1))


def calculate_ap(predicted_keypoints, ground_truth_keypoints, threshold=0.05):
    """
    Calculate Average Precision (AP) for lower body pose estimation.

    Parameters:
        predicted_keypoints (np.array): Predicted keypoints (Nx3: x, y, confidence).
        ground_truth_keypoints (np.array): Ground truth keypoints (Nx3: x, y, visibility).
        threshold (float): The distance threshold to consider a keypoint match.

    Returns:
        float: The Average Precision (AP) value.
    """
    correct_keypoints = 0
    total_keypoints = len(predicted_keypoints)

    for pred, gt in zip(predicted_keypoints, ground_truth_keypoints):
        pred_x, pred_y, pred_conf = pred
        gt_x, gt_y, gt_vis = gt

        # Calculate Euclidean distance between predicted and ground truth keypoint
        distance = np.sqrt((pred_x - gt_x)**2 + (pred_y - gt_y)**2)

        # If the distance is less than the threshold and the ground truth keypoint is visible, it's correct
        if distance < threshold and gt_vis > 0:
            correct_keypoints += 1

    ap = correct_keypoints / total_keypoints
    return ap


def compute_metrics(gt_keypoints, pred_keypoints):
    pck = compute_pck(gt_keypoints, pred_keypoints)
    mpjpe = compute_mpjpe(gt_keypoints, pred_keypoints)
    ap = calculate_ap(gt_keypoints, pred_keypoints)

    print("--- Results ---")
    print(f"PCK: {pck:.2f}%")
    print(f"MPJPE: {mpjpe:.2f} pixels")
    print(f"Average Precision (AP): {ap * 100:.2f}%")

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
    # Convert gt and pred to numpy arrays with the correct dtype
    gt = np.array(gt, dtype=np.float64)
    pred = np.array(pred, dtype=np.float64)

    # Check if the arrays are of the correct shape
    if gt.ndim != 3 or pred.ndim != 3:
        raise ValueError(
            "gt and pred should have shape (N, J, 2) or (N, J, 3)")

    # Define lower-body joint indices for left and right side
    left_hip_idx, left_knee_idx, left_ankle_idx = 0, 1, 2
    right_hip_idx, right_knee_idx, right_ankle_idx = 3, 4, 5

    # Compute thigh and shin lengths for left side (thigh + shin)
    left_thigh_length = np.linalg.norm(
        gt[:, left_knee_idx] - gt[:, left_hip_idx], axis=-1)
    left_shin_length = np.linalg.norm(
        gt[:, left_ankle_idx] - gt[:, left_knee_idx], axis=-1)

    # Compute thigh and shin lengths for right side (thigh + shin)
    right_thigh_length = np.linalg.norm(
        gt[:, right_knee_idx] - gt[:, right_hip_idx], axis=-1)
    right_shin_length = np.linalg.norm(
        gt[:, right_ankle_idx] - gt[:, right_knee_idx], axis=-1)

    # Ensure that all lengths are valid (not NaN or inf)
    if np.any(np.isnan(left_thigh_length)) or np.any(np.isnan(left_shin_length)) or \
       np.any(np.isnan(right_thigh_length)) or np.any(np.isnan(right_shin_length)):
        raise ValueError("NaN values detected in length calculations")

    # Compute lower body lengths (thigh + shin for left and right sides)
    left_lower_body_length = left_thigh_length + left_shin_length
    right_lower_body_length = right_thigh_length + right_shin_length

    # Compute correctness based on threshold
    # For left side: Compare ground truth and predicted keypoints
    left_correct = np.linalg.norm(
        gt[:, left_hip_idx] - pred[:, left_hip_idx], axis=-1) < threshold * left_lower_body_length
    left_correct += np.linalg.norm(gt[:, left_knee_idx] - pred[:,
                                   left_knee_idx], axis=-1) < threshold * left_lower_body_length
    left_correct += np.linalg.norm(gt[:, left_ankle_idx] - pred[:,
                                   left_ankle_idx], axis=-1) < threshold * left_lower_body_length

    # For right side: Compare ground truth and predicted keypoints
    right_correct = np.linalg.norm(
        gt[:, right_hip_idx] - pred[:, right_hip_idx], axis=-1) < threshold * right_lower_body_length
    right_correct += np.linalg.norm(gt[:, right_knee_idx] - pred[:,
                                    right_knee_idx], axis=-1) < threshold * right_lower_body_length
    right_correct += np.linalg.norm(gt[:, right_ankle_idx] - pred[:,
                                    right_ankle_idx], axis=-1) < threshold * right_lower_body_length

    # Combine correctness for both sides
    correct = np.concatenate([left_correct, right_correct], axis=0)

    # Return the mean percentage of correct keypoints
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
    gt = np.array(gt, dtype=np.float64)
    pred = np.array(pred, dtype=np.float64)
    print("Computing MPJPE for lower body...")
    return np.mean(np.linalg.norm(gt - pred, axis=-1))


def calculate_ap_per_joint(predicted_keypoints, ground_truth_keypoints, threshold=50):
    """
    Calculate AP for each joint in lower body pose estimation.

    Parameters:
        predicted_keypoints (np.array): Predicted keypoints (Nx6x2: x, y).
        ground_truth_keypoints (np.array): Ground truth keypoints (Nx6x2: x, y).
        threshold (float): Distance threshold to consider a keypoint as correct.

    Returns:
        dict: AP values for each joint.
    """
    # Convert to NumPy arrays in case input is a list
    predicted_keypoints = np.array(predicted_keypoints)
    ground_truth_keypoints = np.array(ground_truth_keypoints)

    # COCO keypoint mapping (6 joints)
    coco_joints = ['left_hip', 'right_hip', 'left_knee',
                   'right_knee', 'left_ankle', 'right_ankle']

    ap_per_joint = {joint: 0 for joint in coco_joints}  # Store AP per joint
    total_keypoints_per_joint = len(predicted_keypoints)  # Number of frames

    for joint_idx, joint_name in enumerate(coco_joints):
        correct_keypoints = 0

        for pred_frame, gt_frame in zip(predicted_keypoints, ground_truth_keypoints):
            pred_x, pred_y = pred_frame[joint_idx]
            gt_x, gt_y = gt_frame[joint_idx]

            # Compute Euclidean distance
            distance = np.linalg.norm([pred_x - gt_x, pred_y - gt_y])

            # Check if keypoint is within threshold
            if distance < threshold:
                correct_keypoints += 1

        # Compute AP for this joint
        ap_per_joint[joint_name] = correct_keypoints / \
            total_keypoints_per_joint if total_keypoints_per_joint > 0 else 0

    return ap_per_joint


def calculate_map(predicted_keypoints, ground_truth_keypoints, threshold=50):
    """
    Calculate Mean Average Precision (mAP) for lower body keypoints.

    Parameters:
        predicted_keypoints (np.array): Predicted keypoints (Nx6x2: x, y).
        ground_truth_keypoints (np.array): Ground truth keypoints (Nx6x2: x, y).
        threshold (float): Distance threshold to consider a keypoint as correct.

    Returns:
        float: Mean Average Precision (mAP).
    """
    ap_per_joint = calculate_ap_per_joint(
        predicted_keypoints, ground_truth_keypoints, threshold)

    # Compute mean AP (mAP)
    map_value = np.mean(list(ap_per_joint.values()))
    return map_value


def compute_metrics(gt_keypoints, pred_keypoints):
    pck = compute_pck(gt_keypoints, pred_keypoints)
    mpjpe = compute_mpjpe(gt_keypoints, pred_keypoints)
    ap = calculate_ap_per_joint(gt_keypoints, pred_keypoints)
    map_value = calculate_map(pred_keypoints, gt_keypoints)

    print("--- Results ---")
    print(f"PCK: {pck:.2f}%")
    print(f"MPJPE: {mpjpe:.2f} pixels")
    # Assuming ap is a dictionary with AP values for each joint
    for joint, ap_value in ap.items():
        print(
            f"Average Precision (AP) for {joint}: {ap_value:.4f} ({ap_value * 100:.2f}%)")

    # Assuming map_value is a single number (the mAP value)
    print(
        f"Mean Average Precision (mAP): {map_value:.4f} ({map_value * 100:.2f}%)")

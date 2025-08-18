import pickle
import numpy as np
import pandas as pd
import os
from utils.video_io import get_video_resolution, rescale_keypoints


def assess_single_movi_sample(gt_csv_path, pred_pkl_path, video_path=None):
    """
    Load and align MoVi ground truth keypoints and predicted keypoints,
    along with bounding boxes and scores, in a format compatible with HumanEva.

    Parameters:
        gt_csv_path (str): Path to ground truth CSV file (flat array: frames x 2*J).
        pred_pkl_path (str): Path to predicted keypoints in .pkl format.
        video_path (str, optional): Original video path for rescaling (if needed).

    Returns:
        A tuple of aligned lists/arrays:
        (gt_keypoints, gt_bboxes, gt_scores, pred_keypoints, pred_bboxes, pred_scores)
    """
    # --- Load ground truth keypoints, skip header row (joint indices) ---
    df = pd.read_csv(gt_csv_path, header=None, skiprows=1)
    num_joints = df.shape[1] // 2
    gt_keypoints_np = df.values.reshape((-1, num_joints, 2))

    # --- Initialize ground truth bboxes and scores (not available in MoVi) ---
    gt_bboxes = [None] * len(gt_keypoints_np)
    gt_scores = [None] * len(gt_keypoints_np)

    # --- Load predicted keypoints, bboxes, and scores ---
    with open(pred_pkl_path, "rb") as f:
        pred_data = pickle.load(f)

    pred_keypoints = []
    pred_bboxes = []
    pred_scores = []

    for frame in pred_data["keypoints"]:
        people = frame["keypoints"]
        if not people:
            # Handle frames with no detections by appending None
            pred_keypoints.append(None)
            pred_bboxes.append(None)
            pred_scores.append(None)
            continue

        # Assume one person per frame for MoVi dataset
        person = people[0]
        keypoints_arr = np.array(person["keypoints"])
        if keypoints_arr.ndim == 3 and keypoints_arr.shape[0] == 1:
            keypoints_arr = keypoints_arr[0]

        pred_keypoints.append(keypoints_arr)
        pred_bboxes.append(
            np.array(person["bboxes"]) if "bboxes" in person else None)
        pred_scores.append(person.get("scores", None))

    # --- Optional resolution correction ---
    if video_path:
        try:
            orig_w, orig_h = get_video_resolution(video_path)
            test_video_path = pred_pkl_path.replace(".pkl", ".avi")
            if os.path.exists(test_video_path):
                test_w, test_h = get_video_resolution(test_video_path)
                if (test_w, test_h) != (orig_w, orig_h):
                    scale_w = orig_w / test_w
                    scale_h = orig_h / test_h

                    # Rescale keypoints in the list
                    pred_keypoints = [
                        rescale_keypoints(
                            kp, scale_w, scale_h) if kp is not None else None
                        for kp in pred_keypoints
                    ]

                    # Rescale bounding boxes as well
                    pred_bboxes = [
                        [bbox[0] * scale_w, bbox[1] * scale_h,
                            bbox[2] * scale_w, bbox[3] * scale_h]
                        if bbox is not None else None for bbox in pred_bboxes
                    ]
        except Exception:
            pass  # Log or ignore resolution mismatch

    # --- Align ground truth and predictions ---
    min_len = min(len(gt_keypoints_np), len(pred_keypoints))

    gt_keypoints_list = [gt_keypoints_np[i] for i in range(min_len)]
    gt_bboxes = gt_bboxes[:min_len]
    gt_scores = gt_scores[:min_len]

    pred_keypoints = pred_keypoints[:min_len]
    pred_bboxes = pred_bboxes[:min_len]
    pred_scores = pred_scores[:min_len]

    return gt_keypoints_list, gt_bboxes, gt_scores, pred_keypoints, pred_bboxes, pred_scores

import pickle
import numpy as np
import pandas as pd
import os
from utils.video_io import get_video_resolution, rescale_keypoints


def assess_single_movi_sample(gt_csv_path, pred_pkl_path, video_path=None):
    """
    Load and align MoVi ground truth keypoints and predicted keypoints.

    Parameters:
        gt_csv_path (str): Path to ground truth CSV file (flat array: frames x 2*J).
        pred_pkl_path (str): Path to predicted keypoints in .pkl format.
        video_path (str, optional): Original video path for rescaling (if needed).

    Returns:
        (gt_keypoints, pred_keypoints): Tuple of aligned arrays, each (N, J, 2)
    """
    # --- Load ground truth keypoints, skip header row (joint indices) ---
    df = pd.read_csv(gt_csv_path, header=None, skiprows=1)
    num_joints = df.shape[1] // 2
    gt_keypoints = df.values.reshape((-1, num_joints, 2))

    # --- Load predicted keypoints ---
    with open(pred_pkl_path, "rb") as f:
        pred_data = pickle.load(f)

    pred_keypoints = []
    for frame in pred_data["keypoints"]:
        people = frame["keypoints"]
        if not people:
            continue
        keypoints_arr = np.array(people[0]["keypoints"])
        if keypoints_arr.ndim == 3 and keypoints_arr.shape[0] == 1:
            keypoints_arr = keypoints_arr[0]
        pred_keypoints.append(keypoints_arr)

    pred_keypoints = np.stack(pred_keypoints, axis=0)

    # --- Optional resolution correction ---
    if video_path:
        try:
            orig_w, orig_h = get_video_resolution(video_path)
            test_video_path = pred_pkl_path.replace(".pkl", ".avi")
            if os.path.exists(test_video_path):
                test_w, test_h = get_video_resolution(test_video_path)
                if (test_w, test_h) != (orig_w, orig_h):
                    pred_keypoints = rescale_keypoints(
                        pred_keypoints, orig_w / test_w, orig_h / test_h)
        except Exception:
            pass  # optionally log resolution mismatch

    # --- Align ground truth and predictions ---
    min_len = min(len(gt_keypoints), len(pred_keypoints))
    return gt_keypoints[:min_len], pred_keypoints[:min_len]

import pickle
import numpy as np
import pandas as pd
import os
from utils.video_io import get_video_resolution, rescale_keypoints
from evaluation.get_pred_keypoint import PredictionLoader


def assess_single_movi_sample(gt_csv_path, pred_pkl_path, video_path, pipeline_config):
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
    pred_loader = PredictionLoader(
        pred_pkl_path,
        pipeline_config,
    )
    pred_data = pred_loader.load_raw_predictions()
    pred_keypoints, pred_bboxes, pred_scores = pred_loader._extract_keypoints_data(
        pred_data
    )
    # pred_keypoints, pred_bboxes = pred_loader._rescale_predictions(
    #     pred_keypoints, pred_bboxes, video_path
    # )
    # --- Align ground truth and predictions ---
    min_len = min(len(gt_keypoints_np), len(pred_keypoints))

    gt_keypoints_list = [gt_keypoints_np[i] for i in range(min_len)]
    gt_bboxes = gt_bboxes[:min_len]
    gt_scores = gt_scores[:min_len]

    pred_keypoints = pred_keypoints[:min_len]
    pred_bboxes = pred_bboxes[:min_len]
    pred_scores = pred_scores[:min_len]

    return gt_keypoints_list, gt_bboxes, gt_scores, pred_keypoints, pred_bboxes, pred_scores

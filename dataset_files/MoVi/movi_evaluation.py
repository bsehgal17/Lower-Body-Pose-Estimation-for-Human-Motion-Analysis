import os
import logging
import pickle
import numpy as np
import pandas as pd
from config.pipeline_config import PipelineConfig
from config.global_config import GlobalConfig
from utils.import_utils import import_class_from_string
from evaluation.generic_evaluator import MetricsEvaluator, run_assessment
from evaluation.evaluation_registry import EVALUATION_METRICS
from dataset_files.MoVi.movi_assessor import assess_single_movi_sample

logger = logging.getLogger(__name__)


def movi_data_loader(pred_pkl_path, pipeline_config: PipelineConfig, global_config: GlobalConfig):
    """
    Loads and prepares GT and prediction data for a single MoVi sample.
    Returns (gt_keypoints, gt_bboxes, gt_scores, pred_keypoints, pred_bboxes, pred_scores, sample_info).
    This function refactors the data loading logic from run_movi_assessment.
    """
    try:
        # Extract subject ID from the filename
        file = os.path.basename(pred_pkl_path)
        subject_id = file.split("_")[1]
        subject_str = f"Subject_{subject_id}"

        gt_csv_path = os.path.join(
            pipeline_config.paths.ground_truth_file,
            subject_str,
            "joints2d_projected.csv"
        )
        video_filename = file.replace(".pkl", ".avi")
        video_path = os.path.join(
            global_config.paths.input_dir,
            "MoVi",
            "all_cropped_videos",
            video_filename
        )

        # Unpack all six returned values from the assessor function
        gt_keypoints, gt_bboxes, gt_scores, pred_keypoints, pred_bboxes, pred_scores = assess_single_movi_sample(
            gt_csv_path, pred_pkl_path, video_path
        )

        # The original fix to ensure pred_keypoints is a list of arrays
        if isinstance(pred_keypoints, np.ndarray):
            pred_keypoints = [
                pred_keypoints[i] for i in range(pred_keypoints.shape[0])
            ]

        sample_info = {"subject": subject_str}

        return (
            gt_keypoints,
            gt_bboxes,
            gt_scores,
            pred_keypoints,
            pred_bboxes,
            pred_scores,
            sample_info
        )

    except Exception as e:
        logger.error(f"Failed to process {pred_pkl_path}: {e}")
        return None


def run_movi_assessment(
    pipeline_config: PipelineConfig,
    global_config: GlobalConfig,
    output_dir: str,
    input_dir: str,
):
    """
    Main function to run the MoVi dataset assessment, now refactored to use
    the generic run_assessment helper function, similar to HumanEva.
    """
    logger.info("Running MoVi assessment...")

    gt_enum_class = import_class_from_string(
        pipeline_config.dataset.joint_enum_module)
    pred_enum_class = import_class_from_string(
        pipeline_config.dataset.keypoint_format)

    pred_root = pipeline_config.evaluation.input_dir or pipeline_config.detect.output_dir

    # Instantiate the MetricsEvaluator with the output path
    evaluator = MetricsEvaluator(output_path=output_dir)

    # Define the grouping keys
    grouping_keys = ['subject']

    # Use the generic run_assessment to handle file iteration and evaluation
    run_assessment(
        evaluator=evaluator,
        pipeline_config=pipeline_config,
        global_config=global_config,
        input_dir=pred_root,
        output_dir=output_dir,
        gt_enum_class=gt_enum_class,
        pred_enum_class=pred_enum_class,
        data_loader_func=movi_data_loader,  # Pass our new, aligned data loader
        grouping_keys=grouping_keys,  # Pass the grouping keys
    )

    logger.info("MoVi assessment completed.")

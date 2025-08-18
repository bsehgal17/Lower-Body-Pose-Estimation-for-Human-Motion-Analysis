import os
import logging
import pickle
import numpy as np
import pandas as pd
from config.pipeline_config import PipelineConfig
from config.global_config import GlobalConfig
from utils.import_utils import import_class_from_string
from evaluation.generic_evaluator import MetricsEvaluator
from evaluation.evaluation_registry import EVALUATION_METRICS
from dataset_files.MoVi.movi_assessor import assess_single_movi_sample

logger = logging.getLogger(__name__)


def run_movi_assessment(
    pipeline_config: PipelineConfig,
    global_config: GlobalConfig,
    output_dir: str,
    input_dir: str,
):
    """
    Main function to run the MoVi dataset assessment, with custom logic.
    """
    logger.info("Running MoVi assessment...")

    gt_enum_class = import_class_from_string(
        pipeline_config.dataset.joint_enum_module)
    pred_enum_class = import_class_from_string(
        pipeline_config.dataset.keypoint_format)

    pred_root = pipeline_config.evaluation.input_dir or pipeline_config.detect.output_dir

    # Instantiate the MetricsEvaluator with the output path
    output_excel_path = os.path.join(output_dir, "metrics.xlsx")
    evaluator = MetricsEvaluator(output_path=output_excel_path)

    # Define a list to store metadata keys for saving
    grouping_keys = ['subject']

    for root, _, files in os.walk(pred_root):
        for file in files:
            if not file.endswith(".pkl") or "gt" in file:
                continue

            pred_path = os.path.join(root, file)
            subject_id = file.split("_")[1]
            subject_str = f"Subject_{subject_id}"

            try:
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

                gt, pred = assess_single_movi_sample(
                    gt_csv_path, pred_path, video_path
                )
            except Exception as e:
                logger.error(f"Failed to process {file}: {e}")
                continue

            for metric_cfg in pipeline_config.evaluation.metrics:
                metric_name = metric_cfg["name"]
                params = metric_cfg.get("params", {})

                metric_entry = next(
                    (m for m in EVALUATION_METRICS if m["name"] == metric_name), None)
                if not metric_entry:
                    logger.error(f"Metric '{metric_name}' not found.")
                    continue

                calculator = metric_entry["class"](
                    params=params,
                    gt_enum=gt_enum_class,
                    pred_enum=pred_enum_class
                )

                # Create a sample_info dict that the generic evaluate method expects
                sample_info = {"subject": subject_str}

                evaluator.evaluate(
                    calculator,
                    gt,
                    # Placeholder for bboxes and scores (not available in this simplified GT)
                    [None] * len(gt),
                    [None] * len(gt),
                    pred["keypoints"],
                    pred["bboxes"],
                    pred["scores"],
                    sample_info,
                    metric_name,
                    params
                )

    # Final save call, now correctly implemented
    evaluator.save()

    logger.info("MoVi assessment completed.")

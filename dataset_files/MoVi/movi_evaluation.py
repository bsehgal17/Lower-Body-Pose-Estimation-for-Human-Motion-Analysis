import os
import logging
import pandas as pd
import numpy as np
import pickle
from config.pipeline_config import PipelineConfig
from config.global_config import GlobalConfig
from utils.import_utils import import_class_from_string
from evaluation.evaluation_registry import EVALUATION_METRICS
from utils.video_io import get_video_resolution, rescale_keypoints
from dataset_files.MoVi.movi_assessor import assess_single_movi_sample

logger = logging.getLogger(__name__)


class MoViMetricsEvaluator:
    def __init__(self, output_path=None):
        self.results = []
        self.output_path = output_path
        self.added_keys = set()

    def evaluate(self, calculator, gt, pred, subject, metric_name, params):
        result = calculator.compute(gt, pred)

        if isinstance(result, tuple) and len(result) == 2:
            joint_names, jointwise_scores = result
            for joint, scores in zip(joint_names, jointwise_scores.T):
                key = (subject, metric_name, joint, str(params))
                if key not in self.added_keys:
                    self.added_keys.add(key)
                    self.results.append({
                        "subject": subject,
                        "metric": metric_name,
                        "joint": joint,
                        **params,
                        "score": scores.mean(),
                    })
        else:
            key = (subject, metric_name, str(params))
            if key not in self.added_keys:
                self.added_keys.add(key)
                self.results.append({
                    "subject": subject,
                    "metric": metric_name,
                    **params,
                    "score": result,
                })

    def save(self, output_dir: str, parent_folder_name: str):
        if not self.results:
            return

        df = pd.DataFrame(self.results)

        # Save one file per metric
        for metric_name in df["metric"].unique():
            metric_df = df[df["metric"] == metric_name]
            out_path = os.path.join(
                output_dir, f"{parent_folder_name}_{metric_name}.xlsx"
            )
            metric_df.to_excel(out_path, index=False)
            logger.info(f"Saved: {out_path}")


def get_latest_prediction_folder(pred_root: str) -> str:
    candidates = [
        os.path.join(pred_root, d)
        for d in os.listdir(pred_root)
        if os.path.isdir(os.path.join(pred_root, d))
    ]
    if not candidates:
        raise FileNotFoundError(f"No subdirectories found under {pred_root}")
    latest_dir = max(candidates, key=os.path.getmtime)
    logger.info(f"Using latest prediction folder: {latest_dir}")
    return latest_dir


def run_movi_assessment(
    pipeline_config: PipelineConfig,
    global_config: GlobalConfig,
    output_dir: str,
    input_dir: str,
):
    logger.info("Running MoVi assessment...")

    gt_enum_class = import_class_from_string(
        pipeline_config.dataset.joint_enum_module)
    pred_enum_class = import_class_from_string(
        pipeline_config.dataset.keypoint_format)

    base_pred_root = pipeline_config.evaluation.input_dir or pipeline_config.detect.output_dir
    pred_root = get_latest_prediction_folder(base_pred_root)

    evaluator = MoViMetricsEvaluator()

    for root, _, files in os.walk(pred_root):
        for file in files:
            if not file.endswith(".pkl") or "gt" in file:
                continue

            pred_path = os.path.join(root, file)
            # e.g., Subject_3_walking.pkl → '3'
            subject_id = file.split("_")[1]
            subject_str = f"Subject_{subject_id}"

            try:
                # derive ground truth and video paths
                gt_csv_path = os.path.join(
                    pipeline_config.paths.ground_truth_file, subject_str, "joints2d_projected.csv"
                )
                video_filename = file.replace(".pkl", ".avi")
                video_path = os.path.join(
                    global_config.paths.input_dir, "MoVi", "all_cropped_videos", video_filename
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
                    (m for m in EVALUATION_METRICS if m["name"]
                     == metric_name), None
                )
                if not metric_entry:
                    logger.error(f"Metric '{metric_name}' not found.")
                    continue

                expected = set(metric_entry.get("param_spec", []))
                provided = set(params.keys())
                if not provided <= expected:
                    raise ValueError(
                        f"Params for '{metric_name}' do not match. "
                        f"Expected subset of {expected}, got {provided}."
                    )

                calculator = metric_entry["class"](
                    params=params, gt_enum=gt_enum_class, pred_enum=pred_enum_class
                )

                evaluator.evaluate(calculator, gt, pred,
                                   subject_str, metric_name, params)

    parent_folder_name = os.path.basename(os.path.normpath(pred_root))

    evaluator.save(output_dir, parent_folder_name)


logger.info("MoVi assessment completed.")

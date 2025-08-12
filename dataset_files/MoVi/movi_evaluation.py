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
        self.overall_rows = []
        self.jointwise_rows = []
        self.output_path = output_path
        self.joint_names = None

    def evaluate(self, calculator, gt, pred, subject, metric_name, params):
        result = calculator.compute(gt, pred)

        base_row = {
            "subject": subject,
            "metric": metric_name,
        }

        threshold = params.get('threshold', 0)
        key = f"{metric_name}_{threshold:.2f}"

        if isinstance(result, (float, int, np.floating, np.integer)):
            # Overall metric
            overall_row = base_row.copy()
            overall_row[f"overall_{key}"] = result
            self.overall_rows.append(overall_row)

        elif isinstance(result, tuple) and len(result) == 2:
            joint_names, jointwise_scores = result
            if self.joint_names is None:
                self.joint_names = joint_names

            # Jointwise metrics
            jointwise_row = base_row.copy()
            for joint_name, score in zip(joint_names, jointwise_scores):
                jointwise_row[f"{joint_name}_{key}"] = score
            self.jointwise_rows.append(jointwise_row)

    def save(self, output_dir: str, parent_folder_name: str):
        if not (self.overall_rows or self.jointwise_rows):
            return

        # Save overall metrics
        if self.overall_rows:
            df_overall = pd.DataFrame(self.overall_rows)
            df_overall = df_overall.groupby(
                ["subject", "metric"]).first().reset_index()

            overall_path = os.path.join(
                output_dir, f"{parent_folder_name}_overall_metrics.xlsx"
            )

            with pd.ExcelWriter(overall_path, engine='openpyxl') as writer:
                df_overall.to_excel(writer, index=False,
                                    sheet_name='Overall Metrics')

                # Auto-adjust column widths
                worksheet = writer.sheets['Overall Metrics']
                for column in worksheet.columns:
                    col_name = column[0].value
                    max_len = max(
                        df_overall[col_name].astype(str).map(len).max(),
                        len(col_name)
                    ) + 2
                    worksheet.column_dimensions[column[0]
                                                .column_letter].width = max_len

        # Save jointwise metrics
        if self.jointwise_rows:
            df_jointwise = pd.DataFrame(self.jointwise_rows)
            df_jointwise = df_jointwise.groupby(
                ["subject", "metric"]).first().reset_index()

            jointwise_path = os.path.join(
                output_dir, f"{parent_folder_name}_jointwise_metrics.xlsx"
            )

            with pd.ExcelWriter(jointwise_path, engine='openpyxl') as writer:
                df_jointwise.to_excel(
                    writer, index=False, sheet_name='Jointwise Metrics')

                # Auto-adjust column widths
                worksheet = writer.sheets['Jointwise Metrics']
                for column in worksheet.columns:
                    col_name = column[0].value
                    max_len = max(
                        df_jointwise[col_name].astype(str).map(len).max(),
                        len(col_name)
                    ) + 2
                    worksheet.column_dimensions[column[0]
                                                .column_letter].width = max_len


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
                    params=params,
                    gt_enum=gt_enum_class,
                    pred_enum=pred_enum_class
                )

                evaluator.evaluate(
                    calculator,
                    gt,
                    pred,
                    subject_str,
                    metric_name,
                    params
                )

    parent_folder_name = os.path.basename(os.path.normpath(pred_root))
    evaluator.save(output_dir, parent_folder_name)
    logger.info("MoVi assessment completed.")

import os
import pandas as pd
import numpy as np
import logging
from evaluation.evaluation_registry import EVALUATION_METRICS
from openpyxl.utils.dataframe import dataframe_to_rows

logger = logging.getLogger(__name__)


class MetricsEvaluator:
    """
    Generic evaluator to compute and store metrics for any dataset.
    This version is extended to also handle and save per-frame scores.
    """

    def __init__(self, output_path=None):
        self.overall_rows = []
        self.jointwise_rows = []
        self.per_frame_rows = []  # New list for per-frame scores
        self.per_frame_oks_rows = []  # New list for individual OKS scores
        self.output_path = output_path
        self.joint_names = None

    def evaluate(self, calculator, gt, pred, sample_info, metric_name, params):
        """
        Computes a metric and stores the result.

        Args:
            calculator: An instance of a metric calculator.
            gt: Ground truth data.
            pred: Prediction data.
            sample_info (dict): A dictionary with identifying info for the sample
                                (e.g., {"subject": "S1", "action": "Walking"}).
            metric_name (str): The name of the metric.
            params (dict): Parameters used for the metric.
        """
        try:
            result = calculator.compute(gt, pred)
        except Exception as e:
            logger.error(
                f"Error computing metric '{metric_name}' for sample {sample_info}: {e}")
            return

        threshold = params.get('threshold', 0)
        key = f"{metric_name}_{threshold:.2f}"

        if isinstance(result, (float, int, np.floating, np.integer)):
            # Handle a single overall score
            overall_row = sample_info.copy()
            overall_row[f"overall_{key}"] = result
            self.overall_rows.append(overall_row)
        elif isinstance(result, tuple) and len(result) == 2:
            # Handle jointwise scores
            joint_names, jointwise_scores = result
            if self.joint_names is None:
                self.joint_names = joint_names
            jointwise_row = sample_info.copy()
            for joint_name, score in zip(joint_names, jointwise_scores):
                jointwise_row[f"{joint_name}_{key}"] = score
            self.jointwise_rows.append(jointwise_row)
        elif isinstance(result, np.ndarray) and result.ndim == 1:
            # Handle per-frame scores
            num_frames = len(result)
            per_frame_df = pd.DataFrame({
                "frame_idx": np.arange(num_frames),
                f"pck_{key}": result
            })
            for k, v in sample_info.items():
                per_frame_df[k] = v
            self.per_frame_rows.append(per_frame_df)
        elif isinstance(result, dict):
            # Handle dictionary results, like from the MAPCalculator

            # Remove individual_oks_scores as per the user's request
            if 'individual_oks_scores' in result:
                result.pop('individual_oks_scores')

            # Process the rest of the dictionary as overall metrics
            overall_row = sample_info.copy()
            # Add all items from the result dictionary to the overall_row
            for result_key, result_value in result.items():
                if isinstance(result_value, dict):
                    # Flatten nested dictionaries like 'ap_per_threshold'
                    for sub_key, sub_value in result_value.items():
                        overall_row[f"{metric_name}_{sub_key}"] = sub_value
                else:
                    overall_row[f"{metric_name}_{result_key}"] = result_value
            self.overall_rows.append(overall_row)

    def save(self, output_dir, group_keys: list = None):
        if not (self.overall_rows or self.jointwise_rows or self.per_frame_rows):
            logger.warning(
                "No data collected for any metrics. Skipping file save.")
            return

        parent_folder_name = os.path.basename(
            os.path.normpath(self.output_path))
        output_excel_path = os.path.join(
            output_dir, f"{parent_folder_name}_metrics.xlsx")

        with pd.ExcelWriter(output_excel_path, engine='openpyxl') as writer:

            # Save overall metrics
            if self.overall_rows:
                df_overall = pd.DataFrame(self.overall_rows)
                if not df_overall.empty:
                    df_overall = df_overall.groupby(
                        group_keys, dropna=False).first().reset_index()
                df_overall.to_excel(writer, index=False,
                                    sheet_name='Overall Metrics')

                # Add the auto-sizing logic for the 'Overall Metrics' sheet
                worksheet = writer.sheets['Overall Metrics']
                for column in worksheet.columns:
                    col_name = column[0].value
                    # Find the maximum length of the column data or header
                    max_len = max(
                        df_overall[col_name].astype(str).map(len).max(),
                        len(col_name)
                    ) + 2
                    worksheet.column_dimensions[column[0]
                                                .column_letter].width = max_len

            # Save jointwise metrics
            if self.jointwise_rows:
                df_jointwise = pd.DataFrame(self.jointwise_rows)
                if not df_jointwise.empty:
                    df_jointwise = df_jointwise.groupby(
                        group_keys, dropna=False).first().reset_index()
                df_jointwise.to_excel(writer, index=False,
                                      sheet_name='Jointwise Metrics')

                # Add the auto-sizing logic for the 'Jointwise Metrics' sheet
                worksheet = writer.sheets['Jointwise Metrics']
                for column in worksheet.columns:
                    col_name = column[0].value
                    # Find the maximum length of the column data or header
                    max_len = max(
                        df_jointwise[col_name].astype(str).map(len).max(),
                        len(col_name)
                    ) + 2
                    worksheet.column_dimensions[column[0]
                                                .column_letter].width = max_len

            # Save per-frame metrics (no resizing needed for this)
            if self.per_frame_rows:
                # Concatenate all the individual per-frame dataframes
                df_per_frame = pd.concat(
                    self.per_frame_rows, ignore_index=True)

                # Find all unique grouping keys (e.g., subject, action)
                # and all unique metric keys (e.g., pck_per_frame_pck_0.01)
                id_vars = [
                    key for key in group_keys if key in df_per_frame.columns] + ['frame_idx']
                value_vars = [
                    col for col in df_per_frame.columns if col not in id_vars]

                if not id_vars or not value_vars:
                    logger.warning(
                        "Could not identify columns for per-frame score pivoting. Saving as-is.")
                    df_per_frame.to_excel(
                        writer, index=False, sheet_name='Per-Frame Scores')
                else:
                    # Merge scores from different metrics into single rows using a pivot operation
                    # This ensures all scores for a given frame are on the same row.
                    df_per_frame = pd.melt(df_per_frame,
                                           id_vars=id_vars,
                                           value_vars=value_vars,
                                           var_name='metric_id',
                                           value_name='score')

                    df_per_frame = df_per_frame.pivot_table(
                        index=id_vars,
                        columns='metric_id',
                        values='score'
                    ).reset_index()

                    df_per_frame.to_excel(
                        writer, index=False, sheet_name='Per-Frame Scores')


def run_assessment(evaluator, pipeline_config, global_config, input_dir, output_dir, gt_enum_class, pred_enum_class, data_loader_func, group_keys: list = None):
    """
    A generalized function to run evaluation on any dataset.

    Args:
        evaluator (MetricsEvaluator): The generic evaluator instance.
        pipeline_config: Configuration object.
        input_dir (str): Directory containing prediction files.
        output_dir (str): Directory to save output metrics.
        gt_enum_class: The Ground Truth joint enum class.
        pred_enum_class: The Prediction joint enum class.
        data_loader_func (callable): A function that loads GT and pred data for a sample.
                                     It should return (gt, pred, sample_info) or None.
    """
    pred_root = input_dir
    for root, _, files in os.walk(pred_root):
        for file in files:
            if not file.endswith(".pkl") or "gt" in file:
                continue

            pred_pkl_path = os.path.join(root, file)
            sample = data_loader_func(
                pred_pkl_path, pipeline_config, global_config)

            if not sample:
                continue

            gt, pred, sample_info = sample

            for metric_cfg in pipeline_config.evaluation.metrics:
                metric_name = metric_cfg["name"]
                params = metric_cfg.get("params", {})

                metric_entry = next(
                    (m for m in EVALUATION_METRICS if m["name"] == metric_name), None)
                if not metric_entry:
                    logger.error(f"Metric '{metric_name}' not found.")
                    continue

                calculator = metric_entry["class"](
                    params=params, gt_enum=gt_enum_class, pred_enum=pred_enum_class
                )

                evaluator.evaluate(calculator, gt, pred,
                                   sample_info, metric_name, params)

    if evaluator.overall_rows or evaluator.jointwise_rows or evaluator.per_frame_rows or evaluator.per_frame_oks_rows:
        evaluator.output_path = pred_root
        # Pass the received keys to the save method
        evaluator.save(output_dir, group_keys=group_keys)

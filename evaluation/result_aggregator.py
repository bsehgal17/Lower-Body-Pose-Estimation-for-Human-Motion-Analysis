import os
import pandas as pd
import logging
import numpy as np

logger = logging.getLogger(__name__)


class ResultAggregator:
    """
    A unified class to collect and save results from various evaluation metrics.
    """

    def __init__(self, output_path: str):
        self.output_path = output_path
        self.overall_rows = []
        self.jointwise_rows = []
        self.per_frame_rows = []
        self.joint_names = None  # Correctly initialized

        self._current_overall_results = {}
        self._current_jointwise_results = {}
        self._current_per_frame_results = []
        self._current_per_frame_info = {}

    def add_metric(self, sample_info: dict, result, metric_name: str, params: dict):
        """
        Adds a metric result to the appropriate list based on its type.
        This method consolidates results for a single sample before appending.
        """
        sample_key = tuple(sample_info.values())

        # Check if we are processing a new sample
        if (self._current_overall_results is not None and tuple(self._current_overall_results.get(k) for k in sample_info) != sample_key) or \
           (self._current_jointwise_results is not None and tuple(self._current_jointwise_results.get(k) for k in sample_info) != sample_key) or \
           (self._current_per_frame_info is not None and tuple(self._current_per_frame_info.get(k) for k in sample_info) != sample_key):

            # Save and reset the data from the previous sample
            if self._current_overall_results:
                self.overall_rows.append(self._current_overall_results)
            if self._current_jointwise_results:
                self.jointwise_rows.append(self._current_jointwise_results)
            if self._current_per_frame_info:
                self.per_frame_rows.append(
                    pd.DataFrame(self._current_per_frame_results))

            self._current_overall_results = None
            self._current_jointwise_results = None
            self._current_per_frame_results = None
            self._current_per_frame_info = None

        threshold = params.get('threshold', 0)
        key = f"{metric_name}_{threshold:.2f}"

        if isinstance(result, (float, int, np.floating, np.integer)):
            if self._current_overall_results is None:
                self._current_overall_results = sample_info.copy()
            self._current_overall_results[f"overall_{key}"] = result

        elif isinstance(result, tuple) and len(result) == 2:
            if self._current_jointwise_results is None:
                self._current_jointwise_results = sample_info.copy()
            joint_names, jointwise_scores = result
            if self.joint_names is None:
                self.joint_names = joint_names
            for joint_name, score in zip(joint_names, jointwise_scores):
                self._current_jointwise_results[f"{joint_name}_{key}"] = score

        elif isinstance(result, np.ndarray) and result.ndim == 1:
            if self._current_per_frame_results is None:
                self._current_per_frame_results = []
                self._current_per_frame_info = sample_info.copy()

            num_frames = len(result)
            if not self._current_per_frame_results:
                for idx, score in enumerate(result):
                    row = self._current_per_frame_info.copy()
                    row['frame_idx'] = idx
                    row[f"pck_{key}"] = score
                    self._current_per_frame_results.append(row)
            else:
                for idx, score in enumerate(result):
                    self._current_per_frame_results[idx][f"pck_{key}"] = score

        elif isinstance(result, dict):
            if self._current_overall_results is None:
                self._current_overall_results = sample_info.copy()
            if 'individual_oks_scores' in result:
                result.pop('individual_oks_scores')
            for result_key, result_value in result.items():
                if isinstance(result_value, dict):
                    for sub_key, sub_value in result_value.items():
                        self._current_overall_results[f"{metric_name}_{sub_key}"] = sub_value
                else:
                    self._current_overall_results[f"{metric_name}_{result_key}"] = result_value

        elif isinstance(result, np.ndarray) and result.ndim == 1:
            # Check if this is the first per-frame metric added for this sample
            if not self._current_per_frame_results:
                for idx, score in enumerate(result):
                    row = self._current_per_frame_info.copy()
                    row['frame_idx'] = idx
                    row[f"pck_{key}"] = score
                    self._current_per_frame_results.append(row)
            else:
                # Add the new metric to existing per-frame dictionaries
                for idx, score in enumerate(result):
                    self._current_per_frame_results[idx][f"pck_{key}"] = score

        elif isinstance(result, dict):
            if 'individual_oks_scores' in result:
                result.pop('individual_oks_scores')
            for result_key, result_value in result.items():
                if isinstance(result_value, dict):
                    for sub_key, sub_value in result_value.items():
                        self._current_overall_results[f"{metric_name}_{sub_key}"] = sub_value
                else:
                    self._current_overall_results[f"{metric_name}_{result_key}"] = result_value

    def save(self, output_dir, group_keys: list = None):
        """Saves all collected results into a single Excel file with multiple sheets."""

        if self._current_overall_results:
            self.overall_rows.append(self._current_overall_results)
        if self._current_jointwise_results:
            self.jointwise_rows.append(self._current_jointwise_results)
        if self._current_per_frame_results:
            self.per_frame_rows.append(
                pd.DataFrame(self._current_per_frame_results))

        if not (self.overall_rows or self.jointwise_rows or self.per_frame_rows):
            logger.warning(
                "No data collected for any metrics. Skipping file save.")
            return

        if not group_keys:
            group_keys = None

        parent_folder_name = os.path.basename(
            os.path.normpath(self.output_path))
        output_excel_path = os.path.join(
            output_dir, f"{parent_folder_name}_metrics.xlsx")

        output_dir_path = os.path.dirname(output_excel_path)
        if not os.path.exists(output_dir_path):
            os.makedirs(output_dir_path, exist_ok=True)
            logger.info(f"Created directory: {output_dir_path}")

        try:
            with pd.ExcelWriter(output_excel_path, engine='openpyxl') as writer:
                if self.overall_rows:
                    df_overall = pd.DataFrame(self.overall_rows)
                    if not df_overall.empty:
                        df_overall = df_overall.groupby(
                            group_keys, dropna=False).first().reset_index()
                    df_overall.to_excel(writer, index=False,
                                        sheet_name='Overall Metrics')
                    self._autofit_columns(
                        writer, 'Overall Metrics', df_overall)

                if self.jointwise_rows:
                    df_jointwise = pd.DataFrame(self.jointwise_rows)
                    if not df_jointwise.empty:
                        df_jointwise = df_jointwise.groupby(
                            group_keys, dropna=False).first().reset_index()
                    df_jointwise.to_excel(
                        writer, index=False, sheet_name='Jointwise Metrics')
                    self._autofit_columns(
                        writer, 'Jointwise Metrics', df_jointwise)

                if self.per_frame_rows:
                    df_per_frame = pd.concat(
                        self.per_frame_rows, ignore_index=True)
                    df_per_frame.to_excel(
                        writer, index=False, sheet_name='Per-Frame Scores')
                    self._autofit_columns(
                        writer, 'Per-Frame Scores', df_per_frame)

            logger.info(f"Saved results to {output_excel_path}")
        except Exception as e:
            logger.error(f"An error occurred while saving the Excel file: {e}")

    def _autofit_columns(self, writer, sheet_name: str, df: pd.DataFrame):
        """Helper method to auto-size columns in an Excel sheet."""
        worksheet = writer.sheets[sheet_name]
        for column in worksheet.columns:
            col_name = column[0].value
            if col_name in df.columns:
                max_len = max(df[col_name].astype(
                    str).map(len).max(), len(col_name)) + 2
                worksheet.column_dimensions[column[0]
                                            .column_letter].width = max_len

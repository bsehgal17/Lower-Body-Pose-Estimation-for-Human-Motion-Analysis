"""
Data processor replacing original data processing scripts.
"""

from base_classes import BaseDataProcessor
import pandas as pd
import os
from typing import List, Dict, Any
from extractors import MetricExtractorFactory


class DataProcessor(BaseDataProcessor):
    """Unified data processor combining PCK and video data processing."""

    def __init__(self, config):
        super().__init__(config)

    def load_pck_scores(self) -> pd.DataFrame:
        """Load overall PCK scores from Excel file."""
        return self._load_excel_sheet(
            "Overall Metrics", self.config.pck_overall_score_columns
        )

    def load_pck_per_frame_scores(self) -> pd.DataFrame:
        """Load per-frame PCK scores from Excel file."""
        required_cols = ["frame_idx"] + self.config.pck_per_frame_score_columns
        return self._load_excel_sheet("Per-Frame Scores", required_cols)

    def _load_excel_sheet(
        self, sheet_name: str, required_score_columns: List[str]
    ) -> pd.DataFrame:
        """Generic method to load Excel sheets with validation."""
        try:
            df = pd.read_excel(
                self.config.pck_file_path, sheet_name=sheet_name, header=0
            )

            grouping_cols = self.config.get_grouping_columns()

            if grouping_cols:
                df = df.dropna(subset=grouping_cols)
                df = df.reset_index(drop=True)

            required_cols = grouping_cols + required_score_columns
            if not self.validate_data(df, required_cols):
                return None

            if self.config.camera_column and self.config.camera_column in df.columns:
                df[self.config.camera_column] = df[self.config.camera_column].astype(
                    int
                )

            print(f"Successfully loaded {len(df)} records from '{sheet_name}' sheet")
            return df

        except FileNotFoundError:
            print(f"Error: The file {self.config.pck_file_path} was not found.")
            return None
        except Exception as e:
            print(f"An error occurred while loading the '{sheet_name}' sheet: {e}")
            return None

    def process_overall_data(
        self, pck_df: pd.DataFrame, metrics_config: Dict[str, str]
    ) -> Dict[str, Any]:
        """Process video data for overall analysis."""
        results = {}

        for metric_name, extractor_method in metrics_config.items():
            print(f"\nProcessing {metric_name} data...")

            all_metric_data = []
            video_metrics_rows = []

            grouping_cols = self.config.get_grouping_columns()
            if not grouping_cols:
                print(
                    "Warning: No grouping columns found. Cannot perform per-video analysis."
                )
                continue

            grouped_pck_df = pck_df.groupby(grouping_cols)

            for group_name, group_data in grouped_pck_df:
                video_row_data = {
                    col: group_name[grouping_cols.index(col)] for col in grouping_cols
                }
                video_row = pd.Series(video_row_data)

                video_path = self._find_video_for_row(video_row)

                if not video_path or not os.path.exists(video_path):
                    video_info = ", ".join(
                        [f"{k}: {v}" for k, v in video_row_data.items()]
                    )
                    print(f"Warning: Video not found for {video_info}. Skipping.")
                    continue

                extractor = MetricExtractorFactory.create_extractor(
                    metric_name, video_path
                )
                metric_data = extractor.extract()

                if metric_data:
                    synced_start_frame = self._get_synced_start_frame(video_row_data)
                    metric_data_sliced = metric_data[synced_start_frame:]
                    all_metric_data.extend(metric_data_sliced)

                    new_row = video_row_data.copy()
                    new_row[f"avg_{metric_name}"] = pd.Series(metric_data_sliced).mean()
                    video_metrics_rows.append(new_row)

            if video_metrics_rows:
                video_metrics_df = pd.DataFrame(video_metrics_rows)
                merged_df = pd.merge(
                    pck_df, video_metrics_df, on=grouping_cols, how="inner"
                )
                results[metric_name] = {
                    "merged_df": merged_df,
                    "all_metric_data": all_metric_data,
                }
            else:
                print(f"No video data processed for {metric_name}")

        return results

    def process_per_frame_data(
        self, pck_df: pd.DataFrame, metrics_config: Dict[str, str]
    ) -> pd.DataFrame:
        """Process video data for per-frame analysis."""
        combined_rows = []
        grouping_cols = self.config.get_grouping_columns()

        if not grouping_cols:
            print(
                "Warning: No grouping columns found. Cannot perform per-frame analysis."
            )
            return pd.DataFrame()

        grouped_pck_df = pck_df.groupby(grouping_cols)

        for group_name, group_data in grouped_pck_df:
            video_row_data = {
                col: group_name[grouping_cols.index(col)] for col in grouping_cols
            }
            video_row = pd.Series(video_row_data)

            video_path = self._find_video_for_row(video_row)

            if not video_path or not os.path.exists(video_path):
                continue

            video_metrics = {}
            for metric_name, extractor_method in metrics_config.items():
                extractor = MetricExtractorFactory.create_extractor(
                    metric_name, video_path
                )
                metric_data = extractor.extract()

                if metric_data:
                    synced_start_frame = self._get_synced_start_frame(video_row_data)
                    video_metrics[metric_name] = metric_data[synced_start_frame:]

            if video_metrics:
                combined_rows.extend(
                    self._combine_video_metrics_with_pck(
                        group_data, video_metrics, video_row_data
                    )
                )

        return pd.DataFrame(combined_rows) if combined_rows else pd.DataFrame()

    def _find_video_for_row(self, video_row: pd.Series) -> str:
        """Find video file path for a given data row."""
        if self.config.name.lower() == "humaneva":
            return self._find_humaneva_video(video_row)
        elif self.config.name.lower() == "movi":
            return self._find_movi_video(video_row)
        else:
            raise ValueError(f"Unknown dataset: {self.config.name}")

    def _find_humaneva_video(self, video_row: pd.Series) -> str:
        """Find video file for HumanEva dataset."""
        subject = video_row.get(self.config.subject_column)
        action = video_row.get(self.config.action_column)
        camera = video_row.get(self.config.camera_column)

        if not all([subject, action, camera]):
            return None

        video_filename = f"{subject}_{action}_C{camera}.avi"
        video_path = os.path.join(
            self.config.video_directory, subject, action, video_filename
        )

        return video_path if os.path.exists(video_path) else None

    def _find_movi_video(self, video_row: pd.Series) -> str:
        """Find video file for MoVi dataset."""
        subject = video_row.get(self.config.subject_column)

        if not subject:
            return None

        for root, dirs, files in os.walk(self.config.video_directory):
            for file in files:
                if file.endswith(".mp4") and subject in file:
                    return os.path.join(root, file)

        return None

    def _get_synced_start_frame(self, video_row_data: Dict[str, Any]) -> int:
        """Get the synchronized start frame for a video."""
        synced_start_frame = 0

        if hasattr(self.config, "sync_data") and self.config.sync_data:
            try:
                subject_key = str(video_row_data.get(self.config.subject_column))
                action_key = video_row_data.get(self.config.action_column)

                if action_key and isinstance(action_key, str):
                    action_key = action_key.replace("_", " ").title()

                if self.config.camera_column:
                    camera_id = video_row_data.get(self.config.camera_column)
                    camera_index = int(camera_id) - 1
                    synced_start_frame = self.config.sync_data.data[subject_key][
                        action_key
                    ][camera_index]

                if synced_start_frame < 0:
                    synced_start_frame = 0

            except (KeyError, IndexError, TypeError):
                synced_start_frame = 0

        return synced_start_frame

    def _combine_video_metrics_with_pck(
        self,
        pck_data: pd.DataFrame,
        video_metrics: Dict[str, List[float]],
        video_row_data: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Combine video metrics with PCK data on a per-frame basis."""
        combined_rows = []

        for _, pck_row in pck_data.iterrows():
            frame_idx = int(pck_row["frame_idx"])

            row = video_row_data.copy()
            row.update(pck_row.to_dict())

            for metric_name, metric_values in video_metrics.items():
                if frame_idx < len(metric_values):
                    row[metric_name] = metric_values[frame_idx]
                else:
                    row[metric_name] = None

            combined_rows.append(row)

        return combined_rows

    def process(self, *args, **kwargs) -> pd.DataFrame:
        """Main processing method."""
        sheet_type = kwargs.get("sheet_type", "overall")
        if sheet_type == "overall":
            return self.load_pck_scores()
        elif sheet_type == "per_frame":
            return self.load_pck_per_frame_scores()
        else:
            raise ValueError(f"Unknown sheet type: {sheet_type}")

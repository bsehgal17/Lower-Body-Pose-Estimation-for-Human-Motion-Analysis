"""
Data processor using separated components.
"""

from base_classes import BaseDataProcessor
from processors import (
    PCKDataLoader,
    VideoPathResolver,
    FrameSynchronizer,
    DataMerger,
)
from extractors import MetricExtractorFactory
from utils import PerformanceMonitor, ProgressTracker
import pandas as pd
import os
from typing import Dict, Any


class DataProcessor(BaseDataProcessor):
    """Data processor combining all processing functionality."""

    def __init__(self, config):
        super().__init__(config)
        self.pck_loader = PCKDataLoader(config)
        self.path_resolver = VideoPathResolver(config)
        self.frame_sync = FrameSynchronizer(config)
        self.data_merger = DataMerger(config)

    @PerformanceMonitor.timing_decorator
    def load_pck_scores(self) -> pd.DataFrame:
        """Load overall PCK scores from Excel file."""
        return self.pck_loader.load_overall_scores()

    @PerformanceMonitor.timing_decorator
    def load_pck_per_frame_scores(self) -> pd.DataFrame:
        """Load per-frame PCK scores from Excel file."""
        return self.pck_loader.load_per_frame_scores()

    @PerformanceMonitor.timing_decorator
    def process_overall_data(
        self, pck_df: pd.DataFrame, metrics_config: Dict[str, str]
    ) -> Dict[str, Any]:
        """Process video data for overall analysis."""
        results = {}
        grouping_cols = self.config.get_grouping_columns()

        if not grouping_cols:
            print(
                "Warning: No grouping columns found. Cannot perform per-video analysis."
            )
            return results

        grouped_pck_df = pck_df.groupby(grouping_cols)
        progress = ProgressTracker(len(grouped_pck_df), "Processing overall data")

        for metric_name, extractor_method in metrics_config.items():
            print(f"\nProcessing {metric_name} data...")

            all_metric_data = []
            video_metrics_rows = []

            for group_name, group_data in grouped_pck_df:
                video_row_data = {
                    col: group_name[grouping_cols.index(col)] for col in grouping_cols
                }
                video_row = pd.Series(video_row_data)

                video_path = self.path_resolver.find_video_for_row(video_row)

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
                    synced_start_frame = self.frame_sync.get_synced_start_frame(
                        video_row_data
                    )
                    metric_data_sliced = metric_data[synced_start_frame:]
                    all_metric_data.extend(metric_data_sliced)

                    new_row = video_row_data.copy()
                    new_row[f"avg_{metric_name}"] = pd.Series(metric_data_sliced).mean()
                    video_metrics_rows.append(new_row)

                progress.update()

            if video_metrics_rows:
                video_metrics_df = pd.DataFrame(video_metrics_rows)
                merged_df = self.data_merger.merge_overall_data(
                    pck_df, video_metrics_df
                )
                results[metric_name] = {
                    "merged_df": merged_df,
                    "all_metric_data": all_metric_data,
                }
            else:
                print(f"No video data processed for {metric_name}")

        progress.finish()
        return results

    @PerformanceMonitor.timing_decorator
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
        progress = ProgressTracker(len(grouped_pck_df), "Processing per-frame data")

        for group_name, group_data in grouped_pck_df:
            video_row_data = {
                col: group_name[grouping_cols.index(col)] for col in grouping_cols
            }
            video_row = pd.Series(video_row_data)

            video_path = self.path_resolver.find_video_for_row(video_row)

            if not video_path or not os.path.exists(video_path):
                progress.update()
                continue

            video_metrics = {}
            for metric_name, extractor_method in metrics_config.items():
                extractor = MetricExtractorFactory.create_extractor(
                    metric_name, video_path
                )
                metric_data = extractor.extract()

                if metric_data:
                    synced_start_frame = self.frame_sync.get_synced_start_frame(
                        video_row_data
                    )
                    video_metrics[metric_name] = metric_data[synced_start_frame:]

            if video_metrics:
                combined_rows.extend(
                    self.data_merger.combine_video_metrics_with_pck(
                        group_data, video_metrics, video_row_data
                    )
                )

            progress.update()

        progress.finish()
        return pd.DataFrame(combined_rows) if combined_rows else pd.DataFrame()

    def process(self, *args, **kwargs) -> pd.DataFrame:
        """Main processing method."""
        sheet_type = kwargs.get("sheet_type", "overall")
        if sheet_type == "overall":
            return self.load_pck_scores()
        elif sheet_type == "per_frame":
            return self.load_pck_per_frame_scores()
        else:
            raise ValueError(f"Unknown sheet type: {sheet_type}")

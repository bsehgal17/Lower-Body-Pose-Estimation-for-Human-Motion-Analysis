"""
Data merging and combination utilities.
"""

import pandas as pd
from typing import List, Dict, Any
from ..base_classes import BaseDataProcessor


class DataMerger(BaseDataProcessor):
    """Handles merging video metrics with PCK data."""

    def combine_video_metrics_with_pck(
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

    def merge_overall_data(
        self, pck_df: pd.DataFrame, video_metrics_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge PCK data with video metrics for overall analysis."""
        grouping_cols = self.config.get_grouping_columns()
        return pd.merge(pck_df, video_metrics_df, on=grouping_cols, how="inner")

    def process(self, *args, **kwargs) -> pd.DataFrame:
        """Process method required by base class."""
        # This processor doesn't return DataFrames directly
        raise NotImplementedError("Use specific merge methods instead")

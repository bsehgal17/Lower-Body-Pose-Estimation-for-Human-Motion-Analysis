"""
Bin-based statistical analyzer.
"""

from core.base_classes import BaseAnalyzer
import pandas as pd
from typing import Dict, Any
import os


class BinAnalyzer(BaseAnalyzer):
    """Analyzer for bin-based statistical analysis."""

    def analyze(self, data: pd.DataFrame, metric_name: str) -> Dict[str, Any]:
        """Perform bin analysis on the data."""
        print(f"\nAnalyzing Metric: {metric_name.title()}")

        cleaned_data = self._clean_data(data, metric_name)
        binned_data = self._create_bins(cleaned_data, metric_name)
        frame_analysis = self._analyze_frames_per_bin(binned_data, metric_name)
        pck_stats = self._compute_pck_statistics(binned_data, metric_name)
        self._save_results(pck_stats, metric_name)

        return {"frame_analysis": frame_analysis, "pck_statistics": pck_stats}

    def _clean_data(self, df: pd.DataFrame, metric_name: str) -> pd.DataFrame:
        """Clean the data for analysis."""
        cleaned_df = df.copy()

        if cleaned_df.isnull().values.any():
            print("Warning: Missing values found. Filling with 0...")
            columns_to_fill = [metric_name] + self.config.pck_per_frame_score_columns
            cleaned_df.loc[:, columns_to_fill] = cleaned_df.loc[
                :, columns_to_fill
            ].fillna(0)
            cleaned_df.dropna(inplace=True)

        return cleaned_df

    def _create_bins(self, df: pd.DataFrame, metric_name: str) -> pd.DataFrame:
        """Create bins for the metric."""
        if metric_name == "brightness":
            bins = [0, 50, 100, 150, 200, 255]
            labels = ["Low", "Medium-Low", "Medium", "Medium-High", "High"]
        else:
            bins = pd.qcut(df[metric_name], q=5, retbins=True, labels=False)[1]
            labels = [f"Bin {i + 1}" for i in range(5)]

        df[f"{metric_name}_bin"] = pd.cut(
            df[metric_name], bins=bins, labels=labels, right=False
        )
        return df

    def _analyze_frames_per_bin(self, df: pd.DataFrame, metric_name: str) -> pd.Series:
        """Analyze number of frames per bin."""
        bin_column = f"{metric_name}_bin"
        frame_counts = df[bin_column].value_counts()

        print("\nNumber of frames per bin:")
        print(frame_counts)

        return frame_counts

    def _compute_pck_statistics(
        self, df: pd.DataFrame, metric_name: str
    ) -> pd.DataFrame:
        """Compute PCK statistics per bin."""
        bin_column = f"{metric_name}_bin"
        pck_summary_list = []

        for pck_col in self.config.pck_per_frame_score_columns:
            stats = (
                df.groupby(bin_column)[pck_col]
                .agg(mean_pck="mean", median_pck="median", std_pck="std", count="count")
                .reset_index()
            )
            stats["PCK_Column"] = pck_col
            pck_summary_list.append(stats)

        return pd.concat(pck_summary_list, ignore_index=True)

    def _save_results(self, summary_df: pd.DataFrame, metric_name: str):
        """Save bin analysis results."""
        save_path = os.path.join(self.config.save_folder, f"{metric_name}_summary.xlsx")

        os.makedirs(self.config.save_folder, exist_ok=True)

        if os.path.exists(save_path):
            existing_df = pd.read_excel(save_path)
            combined_df = pd.concat([existing_df, summary_df], ignore_index=True)
            combined_df.to_excel(save_path, index=False)
            print(f"Updated summary for {metric_name} saved to '{save_path}'")
        else:
            summary_df.to_excel(save_path, index=False)
            print(f"Summary for {metric_name} created at '{save_path}'")

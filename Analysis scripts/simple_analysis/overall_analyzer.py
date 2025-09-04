"""
Overall Analysis Script

Runs overall analysis on video-level PCK scores and metrics.
Focus: Overall/video-level analysis only.
"""

from visualizers import VisualizationFactory
from core.data_processor import DataProcessor
from config import ConfigManager
import sys
import os
import pandas as pd
from typing import Optional, Dict, Any

# Add the Analysis scripts directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class OverallAnalyzer:
    """Overall analyzer for video-level PCK data."""

    def __init__(self, dataset_name: str):
        """Initialize with dataset name."""
        self.dataset_name = dataset_name
        self.config = ConfigManager.load_config(dataset_name)
        self.data_processor = DataProcessor(self.config)

    def run_overall_analysis(
        self, metrics_config: Dict[str, str] = None
    ) -> Optional[Dict[str, Any]]:
        """Run overall video-level analysis."""
        print(f"Running overall analysis for {self.dataset_name}...")
        print("=" * 50)

        # Default metrics config
        if metrics_config is None:
            metrics_config = {"brightness": "get_brightness_data"}

        # Load overall PCK data
        overall_data = self.data_processor.load_pck_scores()
        if overall_data is None:
            print("❌ Cannot load overall PCK data")
            return None

        print(f"✅ Loaded {len(overall_data)} overall records")

        # Process video metrics
        try:
            print("Processing video metrics...")
            processed_data = self.data_processor.process_overall_data(
                overall_data, metrics_config
            )

            if processed_data:
                print("✅ Overall analysis completed")
                self._print_overall_summary(processed_data)
                return processed_data
            else:
                print("❌ No processed data generated")
                return None

        except Exception as e:
            print(f"❌ Error in overall analysis: {e}")
            return None

    def _print_overall_summary(self, processed_data: Dict[str, Any]):
        """Print overall analysis summary."""
        print("\n📊 Overall Analysis Summary:")
        print("-" * 40)

        for metric_name, metric_data in processed_data.items():
            if "merged_df" not in metric_data:
                continue

            merged_df = metric_data["merged_df"]
            all_metric_data = metric_data["all_metric_data"]

            print(f"\n{metric_name.upper()}:")
            print(f"  • Videos processed: {len(merged_df)}")
            print(f"  • Total frames analyzed: {len(all_metric_data)}")

            # Calculate overall statistics
            if all_metric_data:
                import numpy as np

                mean_val = np.mean(all_metric_data)
                std_val = np.std(all_metric_data)
                print(f"  • Average {metric_name}: {mean_val:.2f} ± {std_val:.2f}")

            # Show PCK score statistics
            pck_columns = [col for col in merged_df.columns if col.startswith("pck")]
            if pck_columns:
                print(f"  • PCK thresholds: {len(pck_columns)}")
                for pck_col in pck_columns[:3]:  # Show first 3
                    if pck_col in merged_df.columns:
                        mean_pck = merged_df[pck_col].mean()
                        print(f"    - {pck_col}: {mean_pck:.2f}%")

    def create_overall_visualizations(
        self, processed_data: Dict[str, Any], save_prefix: str = None
    ) -> bool:
        """Create visualizations for overall analysis."""
        if not processed_data:
            print("❌ No processed data for visualization")
            return False

        if save_prefix is None:
            save_prefix = f"overall_{self.dataset_name}"

        try:
            print("Creating overall analysis visualizations...")

            # Create scatter plots for each metric
            scatter_visualizer = VisualizationFactory.create_visualizer(
                "scatter", self.config
            )
            distribution_visualizer = VisualizationFactory.create_visualizer(
                "distribution", self.config
            )

            for metric_name, metric_data in processed_data.items():
                if "merged_df" not in metric_data:
                    continue

                merged_df = metric_data["merged_df"]

                # Create scatter plots (PCK vs metric)
                scatter_save_path = f"{save_prefix}_{metric_name}_scatter"
                scatter_visualizer.create_plot(
                    merged_df, metric_name, scatter_save_path
                )

                # Create distribution plots
                dist_save_path = f"{save_prefix}_{metric_name}_distribution"
                distribution_visualizer.create_plot(
                    merged_df, metric_name, dist_save_path
                )

            print("✅ Overall visualizations created successfully")
            return True

        except Exception as e:
            print(f"❌ Error creating visualizations: {e}")
            return False

    def export_overall_results(
        self, processed_data: Dict[str, Any], filename: str = None
    ):
        """Export overall analysis results to CSV."""
        if not processed_data:
            print("❌ No data to export")
            return

        if filename is None:
            filename = f"overall_results_{self.dataset_name}.csv"

        # Combine all metrics into one DataFrame
        combined_df = None

        for metric_name, metric_data in processed_data.items():
            if "merged_df" not in metric_data:
                continue

            df = metric_data["merged_df"].copy()

            if combined_df is None:
                combined_df = df
            else:
                # Merge on grouping columns
                grouping_cols = self.config.get_grouping_columns()
                if grouping_cols:
                    # Find metric columns to merge
                    metric_cols = [
                        col
                        for col in df.columns
                        if col.startswith(f"avg_{metric_name}")
                    ]
                    merge_cols = grouping_cols + metric_cols
                    df_subset = df[merge_cols]

                    combined_df = combined_df.merge(
                        df_subset, on=grouping_cols, how="outer"
                    )

        if combined_df is not None:
            output_path = os.path.join(self.config.save_folder, filename)
            os.makedirs(self.config.save_folder, exist_ok=True)
            combined_df.to_csv(output_path, index=False)
            print(f"✅ Overall results exported to: {output_path}")
        else:
            print("❌ No data to export")

    def analyze_pck_correlations(self, processed_data: Dict[str, Any]):
        """Analyze correlations between PCK scores and metrics."""
        if not processed_data:
            return

        print("\n🔍 PCK Correlation Analysis:")
        print("-" * 50)

        for metric_name, metric_data in processed_data.items():
            if "merged_df" not in metric_data:
                continue

            merged_df = metric_data["merged_df"]
            metric_col = f"avg_{metric_name}"

            if metric_col not in merged_df.columns:
                continue

            print(f"\n{metric_name.upper()} correlations:")

            # Find PCK columns
            pck_columns = [col for col in merged_df.columns if col.startswith("pck")]

            for pck_col in pck_columns:
                if pck_col in merged_df.columns:
                    correlation = merged_df[pck_col].corr(merged_df[metric_col])
                    if not pd.isna(correlation):
                        strength = (
                            "strong"
                            if abs(correlation) > 0.7
                            else "moderate"
                            if abs(correlation) > 0.4
                            else "weak"
                        )
                        direction = "positive" if correlation > 0 else "negative"
                        print(
                            f"  {pck_col}: r = {correlation:.3f} ({strength} {direction})"
                        )

    def compare_subjects_or_actions(self, processed_data: Dict[str, Any]):
        """Compare results across subjects or actions."""
        if not processed_data:
            return

        print("\n📊 Group Comparison:")
        print("-" * 50)

        grouping_cols = self.config.get_grouping_columns()
        if not grouping_cols:
            print("No grouping columns available for comparison")
            return

        for metric_name, metric_data in processed_data.items():
            if "merged_df" not in metric_data:
                continue

            merged_df = metric_data["merged_df"]
            metric_col = f"avg_{metric_name}"

            if metric_col not in merged_df.columns:
                continue

            print(f"\n{metric_name.upper()} by groups:")

            # Group by each available grouping column
            for group_col in grouping_cols:
                if group_col in merged_df.columns:
                    group_stats = merged_df.groupby(group_col)[metric_col].agg(
                        ["mean", "std", "count"]
                    )
                    print(f"\n  By {group_col}:")
                    for group_name, row in group_stats.iterrows():
                        print(
                            f"    {group_name}: {row['mean']:.2f} ± {row['std']:.2f} (n={row['count']})"
                        )

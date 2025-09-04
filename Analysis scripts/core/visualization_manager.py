"""
Visualization Manager Module

Handles the creation of various visualizations for analysis results.
"""

import os
from datetime import datetime

# from analyzers import AnalyzerFactory
from visualizers import VisualizationFactory
from utils import ProgressTracker


class VisualizationManager:
    """Manages creation of visualizations for analysis results."""

    def __init__(self, config, timestamp=None):
        """Initialize the visualization manager."""
        self.config = config
        self.timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.viz_factory = VisualizationFactory()

    def create_overall_visualizations(self, merged_df, all_metric_data, metric_name):
        """Create overall analysis visualizations using separated components."""

        # Distribution plots for aggregated metric data
        try:
            hist_viz = self.viz_factory.create_visualizer("histogram", self.config)
            save_path = os.path.join(
                self.config.save_folder,
                f"overall_{metric_name}_histogram_{self.timestamp}.svg",
            )

            # Create a simple DataFrame for visualization
            viz_data = {metric_name: all_metric_data}
            import pandas as pd

            viz_df = pd.DataFrame(viz_data)

            hist_viz.create_plot(viz_df, metric_name, save_path)
        except Exception as e:
            print(f"Warning: Could not create histogram plots for {metric_name}: {e}")

        # Scatter plot using merged data if available
        if (
            merged_df is not None
            and not merged_df.empty
            and f"avg_{metric_name}" in merged_df.columns
        ):
            try:
                print(f"Creating scatter plot for {metric_name}")
                print(f"Merged dataframe columns: {list(merged_df.columns)}")
                print(f"Merged dataframe shape: {merged_df.shape}")

                scatter_viz = self.viz_factory.create_visualizer("scatter", self.config)
                save_path = os.path.join(
                    self.config.save_folder,
                    f"overall_{metric_name}_scatter_{self.timestamp}.svg",
                )

                # Rename the column temporarily for scatter plot
                scatter_df = merged_df.copy()
                scatter_df[metric_name] = scatter_df[f"avg_{metric_name}"]

                print(
                    f"Scatter dataframe columns after rename: {list(scatter_df.columns)}"
                )

                # Check if we have PCK columns
                if hasattr(self.config, "pck_per_frame_score_columns"):
                    available_pck_cols = [
                        col
                        for col in self.config.pck_per_frame_score_columns
                        if col in scatter_df.columns
                    ]
                    print(f"Available PCK columns: {available_pck_cols}")

                    if available_pck_cols:
                        scatter_viz.create_plot(scatter_df, metric_name, save_path)
                    else:
                        print(
                            f"Warning: No PCK columns found in merged data for {metric_name} scatter plot"
                        )
                else:
                    print(
                        "Warning: Config missing pck_per_frame_score_columns attribute"
                    )

            except Exception as e:
                print(f"Warning: Could not create scatter plot for {metric_name}: {e}")
                import traceback

                traceback.print_exc()

    def create_per_frame_visualizations(self, combined_df, metric_name):
        """Create per-frame visualizations using separated components."""

        # Scatter plot
        try:
            scatter_viz = self.viz_factory.create_visualizer("scatter", self.config)
            save_path = os.path.join(
                self.config.save_folder,
                f"per_frame_{metric_name}_scatter_{self.timestamp}.svg",
            )
            scatter_viz.create_plot(combined_df, metric_name, save_path)
        except Exception as e:
            print(f"Warning: Could not create scatter plot for {metric_name}: {e}")

    def create_per_video_visualizations(self, video_aggregated_df, metric_name):
        """Create per-video visualizations using aggregated video data."""
        # Note: Per-video scatter plots are now handled by create_pck_brightness_correlation_plot method
        # to avoid redundancy and provide better correlation-specific visualizations
        pass

    def create_pck_brightness_correlation_plot(
        self,
        combined_df,
        brightness_col="brightness",
        video_id_col="video_id",
        analysis_type="per_frame",
    ):
        """Create correlation plot between average PCK and average brightness per video."""
        try:
            scatter_viz = self.viz_factory.create_visualizer("scatter", self.config)
            save_path = os.path.join(
                self.config.save_folder,
                f"{analysis_type}_pck_brightness_correlation_{self.timestamp}.svg",
            )
            scatter_viz.create_pck_brightness_correlation_plot(
                combined_df, brightness_col, video_id_col, save_path
            )
        except Exception as e:
            print(f"Warning: Could not create PCK vs brightness correlation plot: {e}")
            import traceback

            traceback.print_exc()

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
            dist_viz = self.viz_factory.create_visualizer("distribution", self.config)
            save_path = os.path.join(
                self.config.save_folder,
                f"overall_{metric_name}_distribution_{self.timestamp}.svg",
            )

            # Create a simple DataFrame for visualization
            viz_data = {metric_name: all_metric_data}
            import pandas as pd

            viz_df = pd.DataFrame(viz_data)

            dist_viz.create_plot(viz_df, metric_name, save_path)
        except Exception as e:
            print(
                f"Warning: Could not create distribution plots for {metric_name}: {e}"
            )

        # Bar plot for overall data
        try:
            bar_viz = self.viz_factory.create_visualizer("bar", self.config)
            save_path = os.path.join(
                self.config.save_folder,
                f"overall_{metric_name}_bar_{self.timestamp}.svg",
            )

            # Use the aggregated data for bar plot
            viz_data = {metric_name: all_metric_data}
            import pandas as pd

            viz_df = pd.DataFrame(viz_data)

            bar_viz.create_plot(viz_df, metric_name, save_path)
        except Exception as e:
            print(f"Warning: Could not create bar plot for {metric_name}: {e}")

        # Scatter plot using merged data if available
        if (
            merged_df is not None
            and not merged_df.empty
            and f"avg_{metric_name}" in merged_df.columns
        ):
            try:
                scatter_viz = self.viz_factory.create_visualizer("scatter", self.config)
                save_path = os.path.join(
                    self.config.save_folder,
                    f"overall_{metric_name}_scatter_{self.timestamp}.svg",
                )

                # Rename the column temporarily for scatter plot
                scatter_df = merged_df.copy()
                scatter_df[metric_name] = scatter_df[f"avg_{metric_name}"]

                scatter_viz.create_plot(scatter_df, metric_name, save_path)
            except Exception as e:
                print(f"Warning: Could not create scatter plot for {metric_name}: {e}")

    def create_per_frame_visualizations(self, combined_df, metric_name):
        """Create per-frame visualizations using separated components."""

        # Distribution plots (histogram and box plot)
        try:
            dist_viz = self.viz_factory.create_visualizer("distribution", self.config)
            save_path = os.path.join(
                self.config.save_folder,
                f"per_frame_{metric_name}_distribution_{self.timestamp}.svg",
            )
            dist_viz.create_plot(combined_df, metric_name, save_path)
        except Exception as e:
            print(
                f"Warning: Could not create distribution plots for {metric_name}: {e}"
            )

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

        # Bar plot
        try:
            bar_viz = self.viz_factory.create_visualizer("bar", self.config)
            save_path = os.path.join(
                self.config.save_folder,
                f"per_frame_{metric_name}_bar_{self.timestamp}.svg",
            )
            bar_viz.create_plot(combined_df, metric_name, save_path)
        except Exception as e:
            print(f"Warning: Could not create bar plot for {metric_name}: {e}")

    def create_pck_line_plot(self, combined_df, analysis_type="per_frame"):
        """Create PCK line plot for all thresholds using separated components."""
        try:
            pck_viz = self.viz_factory.create_visualizer("pck_line", self.config)

            # Create a modified save path that includes the analysis type
            modified_save_path = f"{analysis_type}_pck_line_{self.timestamp}"

            pck_viz.create_plot(combined_df, "pck_line", modified_save_path)
            print(f"PCK line plot created for {analysis_type} analysis")

        except Exception as e:
            print(f"Warning: Could not create PCK line plot for {analysis_type}: {e}")
            import traceback

            traceback.print_exc()

    def create_individual_pck_plots(self, combined_df, analysis_type="per_frame"):
        """Create individual line plots for each PCK threshold."""
        try:
            if not hasattr(self.config, "pck_per_frame_score_columns"):
                print("Warning: No PCK score columns found in config")
                return

            import matplotlib.pyplot as plt

            # Define bins (integers from -2 to 102)
            bins = list(range(-2, 102))

            for pck_col in self.config.pck_per_frame_score_columns:
                if pck_col not in combined_df.columns:
                    continue

                plt.figure(figsize=(10, 6))

                # Round values to nearest int and count
                counts = (
                    combined_df[pck_col]
                    .round()
                    .astype(int)
                    .value_counts()
                    .reindex(bins, fill_value=0)
                )

                plt.plot(
                    counts.index,
                    counts.values,
                    marker="o",
                    linestyle="-",
                    linewidth=2,
                    markersize=4,
                )

                # Labels and formatting
                plt.title(f"Frame Count per PCK Score - {pck_col}")
                plt.xlabel("PCK Score")
                plt.ylabel("Number of Frames")
                plt.grid(True, linestyle="--", alpha=0.6)

                # Save individual plot
                save_path = os.path.join(
                    self.config.save_folder,
                    f"{analysis_type}_{pck_col}_line_plot_{self.timestamp}.svg",
                )

                os.makedirs(self.config.save_folder, exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches="tight", format="svg")
                plt.close()

                print(f"Individual PCK line plot saved: {save_path}")

        except Exception as e:
            print(
                f"Warning: Could not create individual PCK plots for {analysis_type}: {e}"
            )
            import traceback

            traceback.print_exc()

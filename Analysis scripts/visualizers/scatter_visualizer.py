"""
Scatter plot visualization components.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from base_classes import BaseVisualizer


class ScatterPlotVisualizer(BaseVisualizer):
    """Visualizer for creating scatter plots."""

    def __init__(self, config):
        """Initialize with configuration."""
        self.config = config

    def create_plot(self, data: pd.DataFrame, metric_name: str, save_path: str):
        """Create separate scatter plots for metric vs each PCK score."""
        if not hasattr(self.config, "pck_per_frame_score_columns"):
            print("Config missing pck_per_frame_score_columns attribute")
            return

        # Extract base path and extension
        base_path = os.path.splitext(save_path)[0]

        # Create separate scatter plot for each PCK column
        for pck_col in self.config.pck_per_frame_score_columns:
            if pck_col in data.columns:
                self._create_single_scatter(
                    data, metric_name, pck_col, f"{base_path}_{pck_col}.svg"
                )

    def _create_single_scatter(
        self, data: pd.DataFrame, metric_name: str, pck_col: str, save_path: str
    ):
        """Create a single scatter plot for metric vs specific PCK column."""
        plt.figure(figsize=(8, 6))

        plt.scatter(data[metric_name], data[pck_col], alpha=0.6, color="blue")
        plt.xlabel(metric_name.title())
        plt.ylabel(pck_col)
        plt.title(f"{metric_name.title()} vs {pck_col}")
        plt.grid(True, alpha=0.3)

        # Save the plot
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight", format="svg")
        plt.close()

        print(f"Scatter plot saved to: {save_path}")

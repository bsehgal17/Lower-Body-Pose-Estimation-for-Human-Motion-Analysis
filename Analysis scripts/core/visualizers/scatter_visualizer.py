"""
Scatter plot visualization components.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from ..base_classes import BaseVisualizer


class ScatterPlotVisualizer(BaseVisualizer):
    """Visualizer for creating scatter plots."""

    def __init__(self, config):
        """Initialize with configuration."""
        self.config = config

    def create_plot(self, data: pd.DataFrame, metric_name: str, save_path: str):
        """Create scatter plot for metric vs PCK scores."""
        if not hasattr(self.config, "pck_per_frame_score_columns"):
            print("Config missing pck_per_frame_score_columns attribute")
            return

        # Create scatter plots for each PCK column
        num_pck_cols = len(self.config.pck_per_frame_score_columns)
        fig, axes = plt.subplots(1, num_pck_cols, figsize=(6 * num_pck_cols, 5))

        if num_pck_cols == 1:
            axes = [axes]

        for i, pck_col in enumerate(self.config.pck_per_frame_score_columns):
            if pck_col in data.columns:
                axes[i].scatter(data[metric_name], data[pck_col], alpha=0.6)
                axes[i].set_xlabel(metric_name.title())
                axes[i].set_ylabel(pck_col)
                axes[i].set_title(f"{metric_name.title()} vs {pck_col}")
                axes[i].grid(True, alpha=0.3)

        plt.tight_layout()

        # Save the plot
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Scatter plot saved to: {save_path}")

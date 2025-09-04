"""
Box plot visualization components.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from core.base_classes import BaseVisualizer


class BoxPlotVisualizer(BaseVisualizer):
    """Visualizer for creating box plots."""

    def __init__(self, config):
        """Initialize with configuration."""
        self.config = config

    def create_plot(self, data: pd.DataFrame, metric_name: str, save_path: str):
        """Create box plot."""
        plt.figure(figsize=(8, 6))

        data.boxplot(column=metric_name)
        plt.title(f"{metric_name.title()} Distribution - Box Plot")
        plt.ylabel(metric_name.title())

        # Save the plot
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight", format="svg")
        plt.close()

        print(f"Box plot saved to: {save_path}")

"""
Histogram visualization components.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from core.base_classes import BaseVisualizer


class HistogramVisualizer(BaseVisualizer):
    """Visualizer for creating histogram plots."""

    def __init__(self, config):
        """Initialize with configuration."""
        self.config = config

    def create_plot(self, data: pd.DataFrame, metric_name: str, save_path: str):
        """Create histogram plot."""
        plt.figure(figsize=(10, 6))

        data[metric_name].hist(bins=30, alpha=0.7, color="skyblue", edgecolor="black")
        plt.title(f"{metric_name.title()} Distribution - Histogram")
        plt.xlabel(metric_name.title())
        plt.ylabel("Frequency")
        plt.grid(True, alpha=0.3)

        # Save the plot
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight", format="svg")
        plt.close()

        print(f"Histogram saved to: {save_path}")

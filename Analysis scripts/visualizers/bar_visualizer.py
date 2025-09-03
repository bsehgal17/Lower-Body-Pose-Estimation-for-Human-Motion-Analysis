"""
Bar plot visualization components.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from base_classes import BaseVisualizer


class BarPlotVisualizer(BaseVisualizer):
    """Visualizer for creating bar plots."""

    def __init__(self, config):
        """Initialize with configuration."""
        self.config = config

    def create_plot(self, data: pd.DataFrame, metric_name: str, save_path: str):
        """Create bar plot for categorical data."""
        # If data is not categorical, create bins first
        if data[metric_name].dtype in ["float64", "int64"]:
            data_binned = pd.cut(data[metric_name], bins=10)
            counts = data_binned.value_counts().sort_index()
        else:
            counts = data[metric_name].value_counts()

        plt.figure(figsize=(12, 6))
        counts.plot(kind="bar", color="lightcoral", alpha=0.8)
        plt.title(f"{metric_name.title()} Distribution - Bar Plot")
        plt.xlabel(metric_name.title())
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save the plot
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Bar plot saved to: {save_path}")

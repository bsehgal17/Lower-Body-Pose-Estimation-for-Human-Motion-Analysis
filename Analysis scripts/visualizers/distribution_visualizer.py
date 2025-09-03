"""
Distribution visualization components.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from ..base_classes import BaseVisualizer


class DistributionVisualizer(BaseVisualizer):
    """Visualizer for creating distribution plots (histograms, box plots)."""

    def __init__(self, config):
        """Initialize with configuration."""
        self.config = config

    def create_plot(self, data: pd.DataFrame, metric_name: str, save_path: str):
        """Create distribution plots for the specified metric."""
        # Create subplot figure
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))

        # Histogram
        data[metric_name].hist(
            bins=30, ax=axes[0], alpha=0.7, color="skyblue", edgecolor="black"
        )
        axes[0].set_title(f"{metric_name.title()} Distribution - Histogram")
        axes[0].set_xlabel(metric_name.title())
        axes[0].set_ylabel("Frequency")
        axes[0].grid(True, alpha=0.3)

        # Box plot
        data.boxplot(column=metric_name, ax=axes[1])
        axes[1].set_title(f"{metric_name.title()} Distribution - Box Plot")
        axes[1].set_ylabel(metric_name.title())

        plt.tight_layout()

        # Save the plot
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Distribution plot saved to: {save_path}")

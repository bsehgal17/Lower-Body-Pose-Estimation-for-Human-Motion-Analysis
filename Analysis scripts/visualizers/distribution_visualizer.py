"""
Distribution visualization components.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from base_classes import BaseVisualizer


class DistributionVisualizer(BaseVisualizer):
    """Visualizer for creating distribution plots (histograms, box plots)."""

    def __init__(self, config):
        """Initialize with configuration."""
        self.config = config

    def create_plot(self, data: pd.DataFrame, metric_name: str, save_path: str):
        """Create separate distribution plots for the specified metric."""
        # Extract base path and extension
        base_path = os.path.splitext(save_path)[0]

        # Create histogram
        self._create_histogram(data, metric_name, f"{base_path}_histogram.svg")

        # Create box plot
        self._create_boxplot(data, metric_name, f"{base_path}_boxplot.svg")

    def _create_histogram(self, data: pd.DataFrame, metric_name: str, save_path: str):
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

    def _create_boxplot(self, data: pd.DataFrame, metric_name: str, save_path: str):
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

"""
Distribution visualization components.
"""

import os
import pandas as pd
from core.base_classes import BaseVisualizer
from .histogram_visualizer import HistogramVisualizer
from .boxplot_visualizer import BoxPlotVisualizer


class DistributionVisualizer(BaseVisualizer):
    """Visualizer for creating distribution plots (histograms, box plots)."""

    def __init__(self, config):
        """Initialize with configuration."""
        self.config = config
        self.histogram_visualizer = HistogramVisualizer(config)
        self.boxplot_visualizer = BoxPlotVisualizer(config)

    def create_plot(self, data: pd.DataFrame, metric_name: str, save_path: str):
        """Create separate distribution plots for the specified metric."""
        # Extract base path and extension
        base_path = os.path.splitext(save_path)[0]

        # Create histogram using dedicated visualizer
        histogram_path = f"{base_path}_histogram.svg"
        self.histogram_visualizer.create_plot(data, metric_name, histogram_path)

        # Create box plot using dedicated visualizer
        boxplot_path = f"{base_path}_boxplot.svg"
        self.boxplot_visualizer.create_plot(data, metric_name, boxplot_path)

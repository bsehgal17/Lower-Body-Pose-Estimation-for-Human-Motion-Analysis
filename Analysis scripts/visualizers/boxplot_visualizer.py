"""
Box plot visualization components.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from core.base_classes import BaseVisualizer
from typing import Union, List, Dict


class BoxPlotVisualizer(BaseVisualizer):
    """Visualizer for creating box plots."""

    def __init__(self, config):
        """Initialize with configuration."""
        self.config = config

    def create_plot(
        self,
        data: Union[pd.DataFrame, List[pd.DataFrame], Dict[str, pd.DataFrame]],
        metric_name: str,
        save_path: str,
    ):
        """
        Create box plot(s) based on input data type.

        Args:
            data: Single DataFrame, list of DataFrames, or dict of DataFrames
            metric_name: Column name to plot
            save_path: Path to save the plot
        """
        if isinstance(data, pd.DataFrame):
            self._create_single_boxplot(data, metric_name, save_path)
        elif isinstance(data, list):
            self._create_multi_boxplot_from_list(data, metric_name, save_path)
        elif isinstance(data, dict):
            self._create_multi_boxplot_from_dict(data, metric_name, save_path)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

    def _create_single_boxplot(
        self, data: pd.DataFrame, metric_name: str, save_path: str
    ):
        """Create single box plot for one dataset."""
        plt.figure(figsize=(8, 6))

        data.boxplot(column=metric_name)
        plt.title(f"{metric_name.title()} Distribution - Box Plot")
        plt.ylabel(metric_name.title())

        # Save the plot
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight", format="svg")
        plt.close()

        print(f"Box plot saved to: {save_path}")

    def _create_multi_boxplot_from_list(
        self, data_list: List[pd.DataFrame], metric_name: str, save_path: str
    ):
        """Create multiple box plots from a list of DataFrames."""
        if not data_list:
            print("Warning: No data provided for box plot")
            return

        plt.figure(figsize=(12, 6))

        # Prepare data for multiple box plots
        data_to_plot = []
        labels = []

        for i, df in enumerate(data_list):
            if metric_name in df.columns:
                data_to_plot.append(df[metric_name].dropna())
                labels.append(f"Dataset {i + 1}")
            else:
                print(f"Warning: Column '{metric_name}' not found in dataset {i + 1}")

        if not data_to_plot:
            print(f"Warning: No valid data found for metric '{metric_name}'")
            return

        # Create box plots
        plt.boxplot(data_to_plot, labels=labels)
        plt.title(f"{metric_name.title()} Distribution Comparison - Box Plots")
        plt.xlabel("Datasets")
        plt.ylabel(metric_name.title())
        plt.grid(True, alpha=0.3)

        # Save the plot
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight", format="svg")
        plt.close()

        print(f"Multi-dataset box plot saved to: {save_path}")

    def _create_multi_boxplot_from_dict(
        self, data_dict: Dict[str, pd.DataFrame], metric_name: str, save_path: str
    ):
        """Create multiple box plots from a dictionary of DataFrames."""
        if not data_dict:
            print("Warning: No data provided for box plot")
            return

        plt.figure(figsize=(12, 6))

        # Prepare data for multiple box plots
        data_to_plot = []
        labels = []

        for label, df in data_dict.items():
            if metric_name in df.columns:
                data_to_plot.append(df[metric_name].dropna())
                labels.append(label)
            else:
                print(f"Warning: Column '{metric_name}' not found in dataset '{label}'")

        if not data_to_plot:
            print(f"Warning: No valid data found for metric '{metric_name}'")
            return

        # Create box plots
        plt.boxplot(data_to_plot, labels=labels)
        plt.title(f"{metric_name.title()} Distribution Comparison - Box Plots")
        plt.xlabel("Datasets")
        plt.ylabel(metric_name.title())
        plt.xticks(rotation=45)  # Rotate labels if they're long
        plt.grid(True, alpha=0.3)

        # Save the plot
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight", format="svg")
        plt.close()

        print(f"Multi-dataset box plot saved to: {save_path}")

    def create_plot_from_files(
        self,
        file_paths: List[str],
        metric_name: str,
        save_path: str,
        file_labels: List[str] = None,
    ):
        """
        Create box plots directly from Excel/CSV files.

        Args:
            file_paths: List of file paths to read
            metric_name: Column name to plot
            save_path: Path to save the plot
            file_labels: Optional labels for each file (defaults to filenames)
        """
        if not file_paths:
            print("Warning: No file paths provided")
            return

        data_dict = {}

        for i, file_path in enumerate(file_paths):
            try:
                # Read file based on extension
                if file_path.endswith(".xlsx") or file_path.endswith(".xls"):
                    df = pd.read_excel(file_path)
                elif file_path.endswith(".csv"):
                    df = pd.read_csv(file_path)
                else:
                    print(f"Warning: Unsupported file format for {file_path}")
                    continue

                # Use provided label or filename
                if file_labels and i < len(file_labels):
                    label = file_labels[i]
                else:
                    label = os.path.basename(file_path).split(".")[0]

                data_dict[label] = df

            except Exception as e:
                print(f"Warning: Could not read file {file_path}: {e}")
                continue

        if data_dict:
            self._create_multi_boxplot_from_dict(data_dict, metric_name, save_path)
        else:
            print("Warning: No valid files could be read")

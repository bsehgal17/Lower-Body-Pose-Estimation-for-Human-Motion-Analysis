"""
Scatter plot visualization components.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from core.base_classes import BaseVisualizer
from typing import List, Dict, Any


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

    def create_pck_brightness_correlation_plot(
        self,
        data: pd.DataFrame,
        brightness_col: str = "brightness",
        video_id_col: str = "video_id",
        save_path: str = None,
    ):
        """
        Create scatter plot of average PCK vs average brightness for all videos.

        Args:
            data: DataFrame containing per-frame data with PCK scores, brightness, and video IDs
            brightness_col: Name of the brightness column
            video_id_col: Name of the video ID column
            save_path: Path to save the plot
        """
        if not hasattr(self.config, "pck_per_frame_score_columns"):
            print("Config missing pck_per_frame_score_columns attribute")
            return

        # Check required columns
        required_cols = [brightness_col, video_id_col]
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            print(f"Warning: Missing required columns: {missing_cols}")
            return

        # Get available PCK columns
        available_pck_cols = [
            col
            for col in self.config.pck_per_frame_score_columns
            if col in data.columns
        ]

        if not available_pck_cols:
            print("Warning: No PCK columns found in data")
            return

        print(
            f"Creating PCK vs Brightness correlation plot for {len(available_pck_cols)} PCK thresholds"
        )

        # Calculate averages per video
        video_averages = []

        for video_id in data[video_id_col].unique():
            video_data = data[data[video_id_col] == video_id]

            if len(video_data) == 0:
                continue

            # Calculate average brightness for this video
            avg_brightness = video_data[brightness_col].mean()

            # Calculate average PCK for each threshold for this video
            for pck_col in available_pck_cols:
                if pck_col in video_data.columns:
                    avg_pck = video_data[pck_col].mean()
                    video_averages.append(
                        {
                            "video_id": video_id,
                            "avg_brightness": avg_brightness,
                            "avg_pck": avg_pck,
                            "pck_threshold": pck_col,
                        }
                    )

        if not video_averages:
            print("Warning: No video averages could be calculated")
            return

        # Convert to DataFrame
        avg_df = pd.DataFrame(video_averages)

        # Create the plot
        plt.figure(figsize=(12, 8))

        # Use different colors for different PCK thresholds
        colors = plt.cm.viridis(np.linspace(0, 1, len(available_pck_cols)))

        for i, pck_col in enumerate(available_pck_cols):
            pck_data = avg_df[avg_df["pck_threshold"] == pck_col]

            plt.scatter(
                pck_data["avg_brightness"],
                pck_data["avg_pck"],
                alpha=0.7,
                color=colors[i],
                label=pck_col,
                s=60,
            )

            # Add trend line
            if len(pck_data) > 1:
                z = np.polyfit(pck_data["avg_brightness"], pck_data["avg_pck"], 1)
                p = np.poly1d(z)
                x_trend = np.linspace(
                    pck_data["avg_brightness"].min(),
                    pck_data["avg_brightness"].max(),
                    100,
                )
                plt.plot(
                    x_trend, p(x_trend), color=colors[i], linestyle="--", alpha=0.8
                )

        plt.xlabel("Average Brightness per Video", fontsize=12)
        plt.ylabel("Average PCK Score per Video", fontsize=12)
        plt.title(
            "Average PCK vs Average Brightness Correlation (All Videos)", fontsize=14
        )
        plt.legend(title="PCK Thresholds", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, alpha=0.3)

        # Add correlation statistics
        self._add_correlation_stats(avg_df, plt.gca())

        plt.tight_layout()

        # Save the plot
        if save_path is None:
            save_path = os.path.join(
                self.config.save_folder, "pck_brightness_correlation.svg"
            )

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight", format="svg")
        plt.close()

        print(f"PCK vs Brightness correlation plot saved to: {save_path}")
        print(
            f"Analyzed {len(avg_df['video_id'].unique())} videos across {len(available_pck_cols)} PCK thresholds"
        )

    def _add_correlation_stats(self, avg_df: pd.DataFrame, ax):
        """Add correlation statistics as text to the plot."""
        stats_text = "Correlation Coefficients:\n"

        for pck_col in avg_df["pck_threshold"].unique():
            pck_data = avg_df[avg_df["pck_threshold"] == pck_col]

            if len(pck_data) > 1:
                corr_coef = np.corrcoef(
                    pck_data["avg_brightness"], pck_data["avg_pck"]
                )[0, 1]
                stats_text += f"{pck_col}: {corr_coef:.3f}\n"

        # Add text box with statistics
        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            fontsize=10,
        )

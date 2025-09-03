"""
PCK line plot visualization components.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from ..base_classes import BaseVisualizer


class PCKLinePlotVisualizer(BaseVisualizer):
    """Visualizer for PCK score line distributions across thresholds."""

    def __init__(self, config):
        """Initialize with configuration."""
        self.config = config

    def create_plot(self, data: pd.DataFrame, metric_name: str, save_path: str):
        """
        Create line plot of PCK score distributions for all thresholds.

        Args:
            data: DataFrame containing per-frame PCK scores
            metric_name: Not used for this visualizer
            save_path: Base path for saving (will be modified)
        """
        print("\n" + "=" * 50)
        print("Running PCK Multi-Threshold Line Plot...")

        if not hasattr(self.config, "pck_per_frame_score_columns"):
            raise ValueError("Config must have pck_per_frame_score_columns attribute.")

        pck_cols = [
            col
            for col in self.config.pck_per_frame_score_columns
            if col in data.columns
        ]
        if not pck_cols:
            print("⚠️ No valid PCK columns found in DataFrame.")
            return

        # Define bins (integers from -2 to 102)
        bins = list(range(-2, 102))

        plt.figure(figsize=(12, 7))

        for pck_col in pck_cols:
            # Round values to nearest int and count
            counts = (
                data[pck_col]
                .round()
                .astype(int)
                .value_counts()
                .reindex(bins, fill_value=0)
            )
            plt.plot(
                counts.index, counts.values, marker="o", linestyle="-", label=pck_col
            )

        # Labels and formatting
        plt.title("Frame Count per PCK Score (All Thresholds)")
        plt.xlabel("PCK Score")
        plt.ylabel("Number of Frames")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(title="Thresholds", bbox_to_anchor=(1.05, 1), loc="upper left")

        # Save with modified path
        final_save_path = os.path.join(
            self.config.save_folder, "pck_all_thresholds.png"
        )
        os.makedirs(self.config.save_folder, exist_ok=True)
        plt.savefig(final_save_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"✅ Multi-threshold PCK line plot saved to {final_save_path}")
        print("=" * 50 + "\nPlot Completed.")

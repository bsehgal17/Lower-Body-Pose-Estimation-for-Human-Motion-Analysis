"""
PCK Brightness Distribution Visualizer.

Creates line plots showing brightness distribution vs normalized frequency
for different PCK scores.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any
from core.base_classes import BaseVisualizer


class PCKBrightnessDistributionVisualizer(BaseVisualizer):
    """Visualizer for PCK brightness distribution analysis."""

    def __init__(self, config):
        """Initialize with configuration."""
        super().__init__(config)

        # Set style for better plots
        plt.style.use("seaborn-v0_8")
        sns.set_palette("husl")

    def create_plot(
        self, analysis_results: Dict[str, Any], save_path: str = None, **kwargs
    ):
        """
        Create brightness frequency plots for PCK brightness distribution.

        Args:
            analysis_results: Results from PCKBrightnessAnalyzer
            save_path: Base path for saving plots
            **kwargs: Additional plotting parameters
        """
        print("\n" + "=" * 50)
        print("Creating PCK Brightness Frequency Plots...")

        if not analysis_results:
            print("No analysis results provided for visualization.")
            return

        # Create brightness frequency plots for each PCK threshold
        for pck_column, results in analysis_results.items():
            if not results or "pck_scores" not in results:
                print(f"Skipping {pck_column} - no valid results")
                continue

            self._create_brightness_frequency_plot(results, pck_column, save_path)

        print("✅ All PCK brightness frequency plots created successfully")
        print("=" * 50)

    def _create_brightness_frequency_plot(
        self, results: Dict[str, Any], pck_column: str, save_path: str
    ):
        """Create line plot of brightness vs normalized frequency for each PCK score."""
        pck_scores = results["pck_scores"]
        brightness_bins_list = results["brightness_bins"]
        normalized_frequencies_list = results["normalized_frequencies"]

        if not pck_scores:
            print(f"No PCK scores found for {pck_column}")
            return

        plt.figure(figsize=(14, 8))

        # Create a colormap for different PCK scores
        colors = plt.cm.viridis(np.linspace(0, 1, len(pck_scores)))

        for i, (pck_score, brightness_bins, norm_freq) in enumerate(
            zip(pck_scores, brightness_bins_list, normalized_frequencies_list)
        ):
            plt.plot(
                brightness_bins,
                norm_freq,
                marker="o",
                linestyle="-",
                linewidth=2,
                markersize=4,
                color=colors[i],
                label=f"PCK {pck_score}",
                alpha=0.8,
            )

        plt.title(
            f"Brightness Distribution by PCK Score\n({pck_column})", fontsize=16, pad=20
        )
        plt.xlabel("Brightness Level", fontsize=14)
        plt.ylabel("Normalized Frequency", fontsize=14)
        plt.grid(True, alpha=0.3, linestyle="--")

        # Customize legend
        plt.legend(
            title="PCK Scores",
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            ncol=1,
            fontsize=10,
        )

        plt.tight_layout()

        # Save plot
        if save_path:
            final_path = os.path.join(
                self.config.save_folder,
                f"{save_path}_{pck_column}_brightness_frequency.svg",
            )
        else:
            final_path = os.path.join(
                self.config.save_folder, f"{pck_column}_brightness_frequency.svg"
            )

        os.makedirs(self.config.save_folder, exist_ok=True)
        plt.savefig(final_path, dpi=300, bbox_inches="tight", format="svg")
        plt.close()

        print(f"✅ Brightness frequency plot saved: {final_path}")

    def create_combined_summary_plot(
        self, analysis_results: Dict[str, Any], save_path: str = None
    ):
        """Create a combined summary plot comparing all PCK thresholds."""
        if not analysis_results:
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("PCK Brightness Distribution Summary", fontsize=16)

        # Plot 1: Average brightness by PCK score for all thresholds
        ax1 = axes[0, 0]
        for pck_column, results in analysis_results.items():
            if "pck_scores" not in results or not results["pck_scores"]:
                continue

            pck_scores = results["pck_scores"]
            means = [results["brightness_stats"][pck]["mean"] for pck in pck_scores]

            ax1.plot(pck_scores, means, "o-", linewidth=2, label=pck_column)

        ax1.set_title("Average Brightness by PCK Score")
        ax1.set_xlabel("PCK Score")
        ax1.set_ylabel("Average Brightness")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Total frame counts by PCK score
        ax2 = axes[0, 1]
        for pck_column, results in analysis_results.items():
            if "pck_scores" not in results or not results["pck_scores"]:
                continue

            pck_scores = results["pck_scores"]
            frame_counts = results["frame_counts"]

            ax2.plot(pck_scores, frame_counts, "o-", linewidth=2, label=pck_column)

        ax2.set_title("Frame Counts by PCK Score")
        ax2.set_xlabel("PCK Score")
        ax2.set_ylabel("Number of Frames")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Brightness standard deviation
        ax3 = axes[1, 0]
        for pck_column, results in analysis_results.items():
            if "pck_scores" not in results or not results["pck_scores"]:
                continue

            pck_scores = results["pck_scores"]
            stds = [results["brightness_stats"][pck]["std"] for pck in pck_scores]

            ax3.plot(pck_scores, stds, "o-", linewidth=2, label=pck_column)

        ax3.set_title("Brightness Std Dev by PCK Score")
        ax3.set_xlabel("PCK Score")
        ax3.set_ylabel("Standard Deviation")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: Hide the fourth subplot or use for additional info
        ax4 = axes[1, 1]
        ax4.axis("off")

        # Add summary text
        summary_text = "Summary Statistics:\n\n"
        for pck_column, results in analysis_results.items():
            if "pck_scores" not in results or not results["pck_scores"]:
                continue

            total_frames = sum(results["frame_counts"])
            unique_pck_scores = len(results["pck_scores"])

            summary_text += f"{pck_column}:\n"
            summary_text += f"  Total Frames: {total_frames}\n"
            summary_text += f"  Unique PCK Scores: {unique_pck_scores}\n\n"

        ax4.text(
            0.1,
            0.9,
            summary_text,
            transform=ax4.transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
        )

        plt.tight_layout()

        # Save combined plot
        if save_path:
            final_path = os.path.join(
                self.config.save_folder, f"{save_path}_pck_brightness_summary.svg"
            )
        else:
            final_path = os.path.join(
                self.config.save_folder, "pck_brightness_summary.svg"
            )

        plt.savefig(final_path, dpi=300, bbox_inches="tight", format="svg")
        plt.close()

        print(f"✅ Combined summary plot saved: {final_path}")

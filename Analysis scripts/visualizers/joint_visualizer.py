"""
Joint Analysis Visualizer

Handles visualization and plotting for joint analysis results.
"""

import matplotlib.pyplot as plt
from typing import Dict
from pathlib import Path


class JointVisualizer:
    """Handles visualization of joint analysis results."""

    def __init__(self, output_dir: Path, save_plots: bool = True):
        """Initialize the visualizer.

        Args:
            output_dir: Directory to save plots
            save_plots: Whether to save plots to files
        """
        self.output_dir = output_dir
        self.save_plots = save_plots

        if save_plots:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_scatter_plot(self, threshold_data: Dict, threshold: float) -> None:
        """Create scatter plot for a specific threshold.

        Args:
            threshold_data: Data for this threshold containing joint info
            threshold: PCK threshold value
        """
        joint_names = threshold_data["joint_names"]
        avg_brightness = threshold_data["avg_brightness"]
        avg_pck = threshold_data["avg_pck"]
        colors = threshold_data["colors"]

        if not joint_names:
            print(f"  WARNING: No data to plot for threshold {threshold}")
            return

        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        fig.suptitle(
            f"Average Brightness vs Average PCK - Threshold {threshold}", fontsize=16
        )

        # Create scatter plot with individual points for legend
        for i, (brightness, pck, label) in enumerate(
            zip(avg_brightness, avg_pck, joint_names)
        ):
            color = colors[i % len(colors)]
            ax.scatter(
                brightness,
                pck,
                c=color,
                alpha=0.8,
                s=100,
                edgecolors="black",
                linewidths=1,
                label=label,
            )

        # Configure plot
        ax.set_xlabel("Average Brightness (LAB L-channel)")
        ax.set_ylabel("Average PCK Score")
        ax.set_title(f"Average Brightness vs Average PCK - Threshold {threshold}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", framealpha=0.9)

        # Add padding to axes for better visibility
        if len(avg_brightness) > 1:
            x_margin = (max(avg_brightness) - min(avg_brightness)) * 0.1
            y_margin = (max(avg_pck) - min(avg_pck)) * 0.1
            ax.set_xlim(min(avg_brightness) - x_margin, max(avg_brightness) + x_margin)
            ax.set_ylim(min(avg_pck) - y_margin, max(avg_pck) + y_margin)

        plt.tight_layout()

        # Save plot
        if self.save_plots:
            plot_file = self.output_dir / f"brightness_pck_threshold_{threshold:g}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches="tight")
            print(f"   Saved plot: {plot_file}")

        plt.close()

    def create_all_visualizations(self, plot_data: Dict[str, Dict]) -> None:
        """Create visualizations for all thresholds.

        Args:
            plot_data: Data organized by threshold for plotting
        """
        print("Creating visualizations...")

        try:
            for threshold, threshold_data in plot_data.items():
                print(f"  Creating plot for threshold {threshold}...")
                self.create_scatter_plot(threshold_data, threshold)

            print("Visualizations completed successfully")

        except Exception as e:
            print(f"ERROR: Error creating visualizations: {e}")
            import traceback

            traceback.print_exc()

    def create_summary_plot(self, plot_data: Dict[str, Dict]) -> None:
        """Create a summary plot showing all thresholds together.

        Args:
            plot_data: Data organized by threshold for plotting
        """
        try:
            # Calculate number of subplots needed
            num_thresholds = len(plot_data)
            if num_thresholds == 0:
                return

            # Create subplots
            fig, axes = plt.subplots(1, num_thresholds, figsize=(6 * num_thresholds, 6))
            if num_thresholds == 1:
                axes = [axes]

            fig.suptitle("Joint Analysis Summary - All Thresholds", fontsize=16)

            for i, (threshold, threshold_data) in enumerate(plot_data.items()):
                ax = axes[i]

                joint_names = threshold_data["joint_names"]
                avg_brightness = threshold_data["avg_brightness"]
                avg_pck = threshold_data["avg_pck"]
                colors = threshold_data["colors"]

                if not joint_names:
                    ax.text(
                        0.5,
                        0.5,
                        f"No data\nfor threshold {threshold}",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                    ax.set_title(f"Threshold {threshold}")
                    continue

                # Create scatter plot
                for j, (brightness, pck, label) in enumerate(
                    zip(avg_brightness, avg_pck, joint_names)
                ):
                    color = colors[j % len(colors)]
                    ax.scatter(
                        brightness,
                        pck,
                        c=color,
                        alpha=0.8,
                        s=60,
                        edgecolors="black",
                        linewidths=0.5,
                        label=label,
                    )

                ax.set_xlabel("Avg Brightness")
                ax.set_ylabel("Avg PCK Score")
                ax.set_title(f"Threshold {threshold}")
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8)

            plt.tight_layout()

            if self.save_plots:
                summary_file = self.output_dir / "joint_analysis_summary.png"
                plt.savefig(summary_file, dpi=300, bbox_inches="tight")
                print(f"   Saved summary plot: {summary_file}")

            plt.close()

        except Exception as e:
            print(f"ERROR: Error creating summary plot: {e}")
            import traceback

            traceback.print_exc()

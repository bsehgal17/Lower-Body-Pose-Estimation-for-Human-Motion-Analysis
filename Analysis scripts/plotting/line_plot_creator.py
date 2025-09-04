"""
Line Plot Creator Script

Creates line plots for brightness vs normalized frequency distributions.
Focus: Line plot visualization only.
"""

import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List

# Add the Analysis scripts directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from distribution_calculator import DistributionCalculator


class LinePlotCreator:
    """Create line plots for PCK brightness distributions."""

    def __init__(self, dataset_name: str):
        """Initialize with dataset name."""
        self.dataset_name = dataset_name
        self.calculator = DistributionCalculator(dataset_name)

        # Set plotting style
        plt.style.use("seaborn-v0_8")
        sns.set_palette("husl")

    def create_frequency_line_plot(
        self,
        target_scores: List[int],
        pck_threshold: str = None,
        bin_size: int = 5,
        save_plot: bool = True,
        filename: str = None,
    ) -> str:
        """Create line plot of brightness vs normalized frequency."""
        print(f"Creating frequency line plot for PCK scores: {target_scores}")

        # Calculate distributions
        distributions = self.calculator.calculate_distributions_for_scores(
            target_scores, pck_threshold, bin_size
        )

        if not distributions:
            print("❌ No distributions available for plotting")
            return ""

        # Create the plot
        plt.figure(figsize=(14, 8))

        # Use different colors for each score
        colors = plt.cm.viridis(np.linspace(0, 1, len(distributions)))

        for i, (score, dist) in enumerate(distributions.items()):
            bin_centers = dist["bin_centers"]
            norm_frequencies = dist["normalized_frequencies"]
            frame_count = dist["total_frames"]

            plt.plot(
                bin_centers,
                norm_frequencies,
                marker="o",
                linestyle="-",
                linewidth=2,
                markersize=4,
                color=colors[i],
                label=f"PCK {score} (n={frame_count})",
                alpha=0.8,
            )

        # Customize plot
        threshold_text = f" ({pck_threshold})" if pck_threshold else ""
        plt.title(
            f"Brightness vs Normalized Frequency{threshold_text}\nDataset: {self.dataset_name}",
            fontsize=16,
            pad=20,
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
        if save_plot:
            if filename is None:
                scores_str = "_".join(map(str, target_scores))
                filename = f"frequency_plot_{self.dataset_name}_scores_{scores_str}.svg"

            from config import ConfigManager

            config = ConfigManager.load_config(self.dataset_name)
            output_path = os.path.join(config.save_folder, filename)
            os.makedirs(config.save_folder, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches="tight", format="svg")
            plt.close()

            print(f"✅ Frequency plot saved to: {output_path}")
            return output_path
        else:
            plt.show()
            return ""

    def create_smoothed_line_plot(
        self,
        target_scores: List[int],
        pck_threshold: str = None,
        bin_size: int = 5,
        smooth_factor: float = 0.3,
        save_plot: bool = True,
        filename: str = None,
    ) -> str:
        """Create smoothed line plot using moving average."""
        print(f"Creating smoothed line plot for PCK scores: {target_scores}")

        # Calculate distributions
        distributions = self.calculator.calculate_distributions_for_scores(
            target_scores, pck_threshold, bin_size
        )

        if not distributions:
            print("❌ No distributions available for plotting")
            return ""

        # Create the plot
        plt.figure(figsize=(14, 8))

        colors = plt.cm.plasma(np.linspace(0, 1, len(distributions)))

        for i, (score, dist) in enumerate(distributions.items()):
            bin_centers = np.array(dist["bin_centers"])
            norm_frequencies = np.array(dist["normalized_frequencies"])

            # Apply smoothing using moving average
            window_size = max(3, int(len(norm_frequencies) * smooth_factor))
            if window_size % 2 == 0:
                window_size += 1  # Ensure odd window size

            if len(norm_frequencies) >= window_size:
                smoothed_freq = np.convolve(
                    norm_frequencies, np.ones(window_size) / window_size, mode="same"
                )
            else:
                smoothed_freq = norm_frequencies

            frame_count = dist["total_frames"]

            # Plot both original and smoothed
            plt.plot(
                bin_centers, norm_frequencies, linestyle=":", alpha=0.4, color=colors[i]
            )

            plt.plot(
                bin_centers,
                smoothed_freq,
                marker="o",
                linestyle="-",
                linewidth=3,
                markersize=5,
                color=colors[i],
                label=f"PCK {score} (n={frame_count})",
                alpha=0.9,
            )

        # Customize plot
        threshold_text = f" ({pck_threshold})" if pck_threshold else ""
        plt.title(
            f"Smoothed Brightness Distribution{threshold_text}\nDataset: {self.dataset_name}",
            fontsize=16,
            pad=20,
        )
        plt.xlabel("Brightness Level", fontsize=14)
        plt.ylabel("Normalized Frequency (Smoothed)", fontsize=14)
        plt.grid(True, alpha=0.3, linestyle="--")

        # Add note about smoothing
        plt.text(
            0.02,
            0.98,
            f"Smoothing factor: {smooth_factor}",
            transform=plt.gca().transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        plt.legend(
            title="PCK Scores", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10
        )

        plt.tight_layout()

        # Save plot
        if save_plot:
            if filename is None:
                scores_str = "_".join(map(str, target_scores))
                filename = f"smoothed_plot_{self.dataset_name}_scores_{scores_str}.svg"

            from config import ConfigManager

            config = ConfigManager.load_config(self.dataset_name)
            output_path = os.path.join(config.save_folder, filename)
            os.makedirs(config.save_folder, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches="tight", format="svg")
            plt.close()

            print(f"✅ Smoothed plot saved to: {output_path}")
            return output_path
        else:
            plt.show()
            return ""

    def create_comparison_plot(
        self,
        score_groups: List[List[int]],
        group_labels: List[str] = None,
        pck_threshold: str = None,
        bin_size: int = 5,
        save_plot: bool = True,
        filename: str = None,
    ) -> str:
        """Create comparison plot for different score groups."""
        print(f"Creating comparison plot for score groups: {score_groups}")

        if group_labels is None:
            group_labels = [f"Group {i + 1}" for i in range(len(score_groups))]

        # Create subplots
        fig, axes = plt.subplots(
            len(score_groups), 1, figsize=(14, 6 * len(score_groups))
        )
        if len(score_groups) == 1:
            axes = [axes]

        for i, (scores, label) in enumerate(zip(score_groups, group_labels)):
            # Calculate distributions for this group
            distributions = self.calculator.calculate_distributions_for_scores(
                scores, pck_threshold, bin_size
            )

            if not distributions:
                axes[i].text(
                    0.5,
                    0.5,
                    f"No data for {label}",
                    transform=axes[i].transAxes,
                    ha="center",
                )
                continue

            # Plot each score in this group
            colors = plt.cm.Set1(np.linspace(0, 1, len(distributions)))

            for j, (score, dist) in enumerate(distributions.items()):
                bin_centers = dist["bin_centers"]
                norm_frequencies = dist["normalized_frequencies"]
                frame_count = dist["total_frames"]

                axes[i].plot(
                    bin_centers,
                    norm_frequencies,
                    marker="o",
                    linestyle="-",
                    linewidth=2,
                    markersize=4,
                    color=colors[j],
                    label=f"PCK {score} (n={frame_count})",
                    alpha=0.8,
                )

            # Customize subplot
            threshold_text = f" ({pck_threshold})" if pck_threshold else ""
            axes[i].set_title(f"{label}{threshold_text}", fontsize=14)
            axes[i].set_xlabel("Brightness Level", fontsize=12)
            axes[i].set_ylabel("Normalized Frequency", fontsize=12)
            axes[i].grid(True, alpha=0.3, linestyle="--")
            axes[i].legend(fontsize=10)

        plt.suptitle(
            f"PCK Score Group Comparison\nDataset: {self.dataset_name}", fontsize=16
        )
        plt.tight_layout()

        # Save plot
        if save_plot:
            if filename is None:
                filename = f"comparison_plot_{self.dataset_name}.svg"

            from config import ConfigManager

            config = ConfigManager.load_config(self.dataset_name)
            output_path = os.path.join(config.save_folder, filename)
            os.makedirs(config.save_folder, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches="tight", format="svg")
            plt.close()

            print(f"✅ Comparison plot saved to: {output_path}")
            return output_path
        else:
            plt.show()
            return ""

    def create_overlay_plot(
        self,
        target_scores: List[int],
        pck_threshold: str = None,
        bin_size: int = 5,
        highlight_score: int = None,
        save_plot: bool = True,
        filename: str = None,
    ) -> str:
        """Create overlay plot with one score highlighted."""
        print(f"Creating overlay plot for PCK scores: {target_scores}")

        # Calculate distributions
        distributions = self.calculator.calculate_distributions_for_scores(
            target_scores, pck_threshold, bin_size
        )

        if not distributions:
            print("❌ No distributions available for plotting")
            return ""

        # Create the plot
        plt.figure(figsize=(14, 8))

        for score, dist in distributions.items():
            bin_centers = dist["bin_centers"]
            norm_frequencies = dist["normalized_frequencies"]
            frame_count = dist["total_frames"]

            # Style based on whether this score is highlighted
            if highlight_score and score == highlight_score:
                # Highlighted score - bold and prominent
                plt.plot(
                    bin_centers,
                    norm_frequencies,
                    marker="o",
                    linestyle="-",
                    linewidth=4,
                    markersize=8,
                    color="red",
                    label=f"PCK {score} (n={frame_count}) ⭐",
                    alpha=1.0,
                    zorder=10,
                )
            else:
                # Background scores - muted
                plt.plot(
                    bin_centers,
                    norm_frequencies,
                    marker="s",
                    linestyle="-",
                    linewidth=1,
                    markersize=3,
                    color="gray",
                    label=f"PCK {score} (n={frame_count})",
                    alpha=0.6,
                    zorder=1,
                )

        # Customize plot
        threshold_text = f" ({pck_threshold})" if pck_threshold else ""
        highlight_text = (
            f"\nHighlighted: PCK {highlight_score}" if highlight_score else ""
        )
        plt.title(
            f"PCK Score Overlay Plot{threshold_text}{highlight_text}\nDataset: {self.dataset_name}",
            fontsize=16,
            pad=20,
        )
        plt.xlabel("Brightness Level", fontsize=14)
        plt.ylabel("Normalized Frequency", fontsize=14)
        plt.grid(True, alpha=0.3, linestyle="--")

        plt.legend(
            title="PCK Scores", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10
        )

        plt.tight_layout()

        # Save plot
        if save_plot:
            if filename is None:
                scores_str = "_".join(map(str, target_scores))
                highlight_str = (
                    f"_highlight_{highlight_score}" if highlight_score else ""
                )
                filename = f"overlay_plot_{self.dataset_name}_scores_{scores_str}{highlight_str}.svg"

            from config import ConfigManager

            config = ConfigManager.load_config(self.dataset_name)
            output_path = os.path.join(config.save_folder, filename)
            os.makedirs(config.save_folder, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches="tight", format="svg")
            plt.close()

            print(f"✅ Overlay plot saved to: {output_path}")
            return output_path
        else:
            plt.show()
            return ""


def main():
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Line Plot Creator")
    parser.add_argument("dataset", help="Dataset name (e.g., 'movi', 'humaneva')")
    parser.add_argument("scores", nargs="+", type=int, help="PCK scores to plot")
    parser.add_argument("--threshold", help="Specific PCK threshold to analyze")
    parser.add_argument("--bin-size", type=int, default=5, help="Brightness bin size")
    parser.add_argument(
        "--type",
        choices=["frequency", "smoothed", "overlay"],
        default="frequency",
        help="Type of line plot",
    )
    parser.add_argument(
        "--smooth-factor",
        type=float,
        default=0.3,
        help="Smoothing factor for smoothed plots",
    )
    parser.add_argument(
        "--highlight", type=int, help="Score to highlight in overlay plot"
    )
    parser.add_argument("--filename", help="Custom filename for saved plot")
    parser.add_argument(
        "--show", action="store_true", help="Show plot instead of saving"
    )

    args = parser.parse_args()

    try:
        creator = LinePlotCreator(args.dataset)

        save_plot = not args.show

        if args.type == "frequency":
            creator.create_frequency_line_plot(
                args.scores, args.threshold, args.bin_size, save_plot, args.filename
            )

        elif args.type == "smoothed":
            creator.create_smoothed_line_plot(
                args.scores,
                args.threshold,
                args.bin_size,
                args.smooth_factor,
                save_plot,
                args.filename,
            )

        elif args.type == "overlay":
            creator.create_overlay_plot(
                args.scores,
                args.threshold,
                args.bin_size,
                args.highlight,
                save_plot,
                args.filename,
            )

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

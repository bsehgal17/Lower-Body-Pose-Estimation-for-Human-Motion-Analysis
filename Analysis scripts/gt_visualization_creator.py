"""
Ground Truth Visualization Creator Script

Creates plots and visualizations for ground truth PCK-brightness analysis.
Focus: Visualization creation for GT analysis only.
"""

import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict

# Add the Analysis scripts directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gt_distribution_calculator import GTDistributionCalculator


class GTVisualizationCreator:
    """Create visualizations for ground truth analysis."""

    def __init__(self, dataset_name: str):
        """Initialize with dataset name."""
        self.dataset_name = dataset_name
        self.calculator = GTDistributionCalculator(dataset_name)

        # Set plotting style
        plt.style.use("seaborn-v0_8")
        sns.set_palette("husl")

    def create_pck_brightness_line_plot(
        self,
        pck_brightness_distributions: Dict,
        pck_threshold: str = None,
        target_scores: List[int] = None,
        save_plot: bool = True,
        filename: str = None,
    ) -> str:
        """Create line plot of PCK scores vs brightness distributions."""
        if pck_threshold is None:
            pck_threshold = list(pck_brightness_distributions.keys())[0]

        if pck_threshold not in pck_brightness_distributions:
            print(f"❌ PCK threshold {pck_threshold} not found")
            return ""

        distributions = pck_brightness_distributions[pck_threshold]

        if target_scores:
            distributions = {
                k: v for k, v in distributions.items() if k in target_scores
            }

        if not distributions:
            print("❌ No distributions available for plotting")
            return ""

        print(f"Creating PCK-brightness line plot for {len(distributions)} PCK scores")

        # Create the plot
        plt.figure(figsize=(14, 8))

        colors = plt.cm.viridis(np.linspace(0, 1, len(distributions)))

        for i, (pck_score, dist) in enumerate(distributions.items()):
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
                label=f"PCK {pck_score} (n={frame_count})",
                alpha=0.8,
            )

        # Customize plot
        plt.title(
            f"GT Joint Brightness vs Normalized Frequency\nPCK Threshold: {pck_threshold}, Dataset: {self.dataset_name}",
            fontsize=16,
            pad=20,
        )
        plt.xlabel("Brightness Level at GT Joint Coordinates", fontsize=14)
        plt.ylabel("Normalized Frequency", fontsize=14)
        plt.grid(True, alpha=0.3, linestyle="--")

        plt.legend(
            title="PCK Scores", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10
        )

        plt.tight_layout()

        # Save plot
        if save_plot:
            if filename is None:
                scores_str = (
                    "_".join(map(str, target_scores)) if target_scores else "all_scores"
                )
                filename = f"gt_pck_brightness_plot_{self.dataset_name}_{pck_threshold}_{scores_str}.svg"

            from config import ConfigManager

            config = ConfigManager.load_config(self.dataset_name)
            output_path = os.path.join(config.save_folder, filename)
            os.makedirs(config.save_folder, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches="tight", format="svg")
            plt.close()

            print(f"✅ GT PCK-brightness plot saved to: {output_path}")
            return output_path
        else:
            plt.show()
            return ""

    def create_joint_brightness_comparison_plot(
        self,
        joint_brightness_distributions: Dict,
        target_joints: List[str] = None,
        save_plot: bool = True,
        filename: str = None,
    ) -> str:
        """Create comparison plot of brightness distributions across joints."""
        if target_joints:
            distributions = {
                k: v
                for k, v in joint_brightness_distributions.items()
                if k in target_joints
            }
        else:
            distributions = joint_brightness_distributions

        if not distributions:
            print("❌ No joint distributions available for plotting")
            return ""

        print(
            f"Creating joint brightness comparison plot for {len(distributions)} joints"
        )

        # Create the plot
        plt.figure(figsize=(14, 8))

        colors = plt.cm.Set1(np.linspace(0, 1, len(distributions)))

        for i, (joint_name, dist) in enumerate(distributions.items()):
            bin_centers = dist["bin_centers"]
            norm_frequencies = dist["normalized_frequencies"]
            frame_count = dist["total_frames"]

            plt.plot(
                bin_centers,
                norm_frequencies,
                marker="s",
                linestyle="-",
                linewidth=2,
                markersize=4,
                color=colors[i],
                label=f"{joint_name} (n={frame_count})",
                alpha=0.8,
            )

        # Customize plot
        plt.title(
            f"Ground Truth Joint Brightness Comparison\nDataset: {self.dataset_name}",
            fontsize=16,
            pad=20,
        )
        plt.xlabel("Brightness Level", fontsize=14)
        plt.ylabel("Normalized Frequency", fontsize=14)
        plt.grid(True, alpha=0.3, linestyle="--")

        plt.legend(
            title="Joints", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10
        )

        plt.tight_layout()

        # Save plot
        if save_plot:
            if filename is None:
                joints_str = (
                    "_".join(target_joints[:3])
                    if target_joints and len(target_joints) <= 3
                    else f"{len(distributions)}_joints"
                )
                filename = f"gt_joint_brightness_comparison_{self.dataset_name}_{joints_str}.svg"

            from config import ConfigManager

            config = ConfigManager.load_config(self.dataset_name)
            output_path = os.path.join(config.save_folder, filename)
            os.makedirs(config.save_folder, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches="tight", format="svg")
            plt.close()

            print(f"✅ Joint brightness comparison plot saved to: {output_path}")
            return output_path
        else:
            plt.show()
            return ""

    def create_correlation_heatmap(
        self, analysis_results: Dict, save_plot: bool = True, filename: str = None
    ) -> str:
        """Create heatmap of PCK-brightness correlations."""
        if "correlations" not in analysis_results:
            print("❌ No correlation data available")
            return ""

        correlations = analysis_results["correlations"]

        # Prepare data for heatmap
        joints = list(correlations.keys())
        thresholds = set()
        for joint_corrs in correlations.values():
            thresholds.update(joint_corrs.keys())
        thresholds = sorted(list(thresholds))

        if not joints or not thresholds:
            print("❌ Insufficient correlation data for heatmap")
            return ""

        print(
            f"Creating correlation heatmap for {len(joints)} joints and {len(thresholds)} thresholds"
        )

        # Create correlation matrix
        corr_matrix = np.zeros((len(joints), len(thresholds)))

        for i, joint in enumerate(joints):
            for j, threshold in enumerate(thresholds):
                if threshold in correlations[joint]:
                    corr_matrix[i, j] = correlations[joint][threshold]["correlation"]
                else:
                    corr_matrix[i, j] = np.nan

        # Create the heatmap
        plt.figure(figsize=(max(8, len(thresholds) * 1.5), max(6, len(joints) * 0.8)))

        mask = np.isnan(corr_matrix)

        sns.heatmap(
            corr_matrix,
            xticklabels=thresholds,
            yticklabels=joints,
            cmap="RdBu_r",
            center=0,
            vmin=-1,
            vmax=1,
            mask=mask,
            annot=True,
            fmt=".3f",
            square=False,
            cbar_kws={"label": "Correlation Coefficient"},
        )

        plt.title(
            f"PCK-Brightness Correlation Heatmap\nDataset: {self.dataset_name}",
            fontsize=16,
            pad=20,
        )
        plt.xlabel("PCK Threshold", fontsize=14)
        plt.ylabel("Joint", fontsize=14)

        # Rotate x-axis labels if needed
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)

        plt.tight_layout()

        # Save plot
        if save_plot:
            if filename is None:
                filename = f"gt_correlation_heatmap_{self.dataset_name}.svg"

            from config import ConfigManager

            config = ConfigManager.load_config(self.dataset_name)
            output_path = os.path.join(config.save_folder, filename)
            os.makedirs(config.save_folder, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches="tight", format="svg")
            plt.close()

            print(f"✅ Correlation heatmap saved to: {output_path}")
            return output_path
        else:
            plt.show()
            return ""

    def create_pck_score_distribution_plot(
        self,
        pck_score_distributions: Dict,
        save_plot: bool = True,
        filename: str = None,
    ) -> str:
        """Create bar plot of PCK score distributions."""
        if not pck_score_distributions:
            print("❌ No PCK score distributions available")
            return ""

        print(
            f"Creating PCK score distribution plot for {len(pck_score_distributions)} thresholds"
        )

        # Create subplots for each threshold
        fig, axes = plt.subplots(
            len(pck_score_distributions),
            1,
            figsize=(12, 4 * len(pck_score_distributions)),
        )

        if len(pck_score_distributions) == 1:
            axes = [axes]

        for i, (threshold, dist) in enumerate(pck_score_distributions.items()):
            scores = dist["unique_scores"]
            frequencies = dist["normalized_frequencies"]

            bars = axes[i].bar(
                scores,
                frequencies,
                alpha=0.7,
                color=plt.cm.viridis(i / len(pck_score_distributions)),
            )

            axes[i].set_title(f"PCK Score Distribution - {threshold}", fontsize=14)
            axes[i].set_xlabel("PCK Score", fontsize=12)
            axes[i].set_ylabel("Normalized Frequency", fontsize=12)
            axes[i].grid(True, alpha=0.3, axis="y")

            # Add value labels on bars
            for bar, freq in zip(bars, frequencies):
                if freq > 0:
                    axes[i].text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        f"{freq:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=10,
                    )

        plt.suptitle(
            f"Ground Truth PCK Score Distributions\nDataset: {self.dataset_name}",
            fontsize=16,
        )
        plt.tight_layout()

        # Save plot
        if save_plot:
            if filename is None:
                filename = f"gt_pck_score_distributions_{self.dataset_name}.svg"

            from config import ConfigManager

            config = ConfigManager.load_config(self.dataset_name)
            output_path = os.path.join(config.save_folder, filename)
            os.makedirs(config.save_folder, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches="tight", format="svg")
            plt.close()

            print(f"✅ PCK score distribution plot saved to: {output_path}")
            return output_path
        else:
            plt.show()
            return ""

    def create_brightness_statistics_plot(
        self,
        joint_brightness_distributions: Dict,
        save_plot: bool = True,
        filename: str = None,
    ) -> str:
        """Create box plot showing brightness statistics across joints."""
        if not joint_brightness_distributions:
            print("❌ No joint brightness distributions available")
            return ""

        print(
            f"Creating brightness statistics plot for {len(joint_brightness_distributions)} joints"
        )

        # Prepare data for box plot
        joint_names = []
        brightness_data = []

        for joint_name, dist in joint_brightness_distributions.items():
            # Reconstruct brightness values from distribution
            bin_centers = dist["bin_centers"]
            frequencies = dist["frequencies"]

            joint_brightness = []
            for brightness, freq in zip(bin_centers, frequencies):
                joint_brightness.extend([brightness] * freq)

            if joint_brightness:
                joint_names.append(joint_name)
                brightness_data.append(joint_brightness)

        if not brightness_data:
            print("❌ No brightness data for statistics plot")
            return ""

        # Create the plot
        plt.figure(figsize=(max(10, len(joint_names) * 1.2), 8))

        # Create box plot
        box_plot = plt.boxplot(brightness_data, labels=joint_names, patch_artist=True)

        # Customize colors
        colors = plt.cm.Set3(np.linspace(0, 1, len(joint_names)))
        for patch, color in zip(box_plot["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        plt.title(
            f"Ground Truth Joint Brightness Statistics\nDataset: {self.dataset_name}",
            fontsize=16,
            pad=20,
        )
        plt.xlabel("Joint", fontsize=14)
        plt.ylabel("Brightness Level", fontsize=14)
        plt.xticks(rotation=45, ha="right")
        plt.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()

        # Save plot
        if save_plot:
            if filename is None:
                filename = f"gt_brightness_statistics_{self.dataset_name}.svg"

            from config import ConfigManager

            config = ConfigManager.load_config(self.dataset_name)
            output_path = os.path.join(config.save_folder, filename)
            os.makedirs(config.save_folder, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches="tight", format="svg")
            plt.close()

            print(f"✅ Brightness statistics plot saved to: {output_path}")
            return output_path
        else:
            plt.show()
            return ""

    def create_comprehensive_gt_analysis_dashboard(
        self, distribution_results: Dict, save_plot: bool = True, filename: str = None
    ) -> str:
        """Create comprehensive dashboard with multiple GT analysis plots."""
        print("Creating comprehensive GT analysis dashboard")

        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))

        # Define layout
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], hspace=0.3, wspace=0.3)

        try:
            # 1. PCK-Brightness line plot (top left)
            ax1 = fig.add_subplot(gs[0, 0])
            plt.sca(ax1)

            pck_brightness_dists = distribution_results.get(
                "pck_brightness_distributions", {}
            )
            if pck_brightness_dists:
                threshold = list(pck_brightness_dists.keys())[0]
                distributions = pck_brightness_dists[threshold]

                colors = plt.cm.viridis(np.linspace(0, 1, len(distributions)))
                for i, (pck_score, dist) in enumerate(distributions.items()):
                    plt.plot(
                        dist["bin_centers"],
                        dist["normalized_frequencies"],
                        marker="o",
                        linestyle="-",
                        color=colors[i],
                        label=f"PCK {pck_score}",
                        alpha=0.8,
                    )

                plt.title("PCK-Brightness Distributions", fontsize=12)
                plt.xlabel("Brightness Level")
                plt.ylabel("Normalized Frequency")
                plt.legend(fontsize=8)
                plt.grid(True, alpha=0.3)

            # 2. Joint comparison (top right)
            ax2 = fig.add_subplot(gs[0, 1])
            plt.sca(ax2)

            joint_dists = distribution_results.get("joint_brightness_distributions", {})
            if joint_dists:
                colors = plt.cm.Set1(np.linspace(0, 1, len(joint_dists)))
                for i, (joint_name, dist) in enumerate(joint_dists.items()):
                    plt.plot(
                        dist["bin_centers"],
                        dist["normalized_frequencies"],
                        marker="s",
                        linestyle="-",
                        color=colors[i],
                        label=joint_name,
                        alpha=0.8,
                    )

                plt.title("Joint Brightness Comparison", fontsize=12)
                plt.xlabel("Brightness Level")
                plt.ylabel("Normalized Frequency")
                plt.legend(fontsize=8)
                plt.grid(True, alpha=0.3)

            # 3. Correlation heatmap (middle)
            ax3 = fig.add_subplot(gs[1, :])
            plt.sca(ax3)

            analysis_results = distribution_results.get("analysis_results", {})
            if "correlations" in analysis_results:
                correlations = analysis_results["correlations"]

                joints = list(correlations.keys())
                thresholds = set()
                for joint_corrs in correlations.values():
                    thresholds.update(joint_corrs.keys())
                thresholds = sorted(list(thresholds))

                if joints and thresholds:
                    corr_matrix = np.zeros((len(joints), len(thresholds)))

                    for i, joint in enumerate(joints):
                        for j, threshold in enumerate(thresholds):
                            if threshold in correlations[joint]:
                                corr_matrix[i, j] = correlations[joint][threshold][
                                    "correlation"
                                ]
                            else:
                                corr_matrix[i, j] = np.nan

                    mask = np.isnan(corr_matrix)
                    sns.heatmap(
                        corr_matrix,
                        xticklabels=thresholds,
                        yticklabels=joints,
                        cmap="RdBu_r",
                        center=0,
                        vmin=-1,
                        vmax=1,
                        mask=mask,
                        annot=True,
                        fmt=".2f",
                        cbar_kws={"label": "Correlation"},
                    )

                    plt.title("PCK-Brightness Correlations", fontsize=12)

            # 4. PCK distributions (bottom)
            ax4 = fig.add_subplot(gs[2, :])
            plt.sca(ax4)

            pck_dists = distribution_results.get("pck_score_distributions", {})
            if pck_dists:
                # Combine all PCK distributions
                all_scores = set()
                for dist in pck_dists.values():
                    all_scores.update(dist["unique_scores"])
                all_scores = sorted(list(all_scores))

                x_pos = np.arange(len(all_scores))
                width = 0.8 / len(pck_dists)

                for i, (threshold, dist) in enumerate(pck_dists.items()):
                    frequencies = []
                    for score in all_scores:
                        if score in dist["unique_scores"]:
                            idx = dist["unique_scores"].index(score)
                            frequencies.append(dist["normalized_frequencies"][idx])
                        else:
                            frequencies.append(0)

                    plt.bar(
                        x_pos + i * width,
                        frequencies,
                        width,
                        label=threshold,
                        alpha=0.7,
                    )

                plt.title("PCK Score Distributions", fontsize=12)
                plt.xlabel("PCK Score")
                plt.ylabel("Normalized Frequency")
                plt.xticks(x_pos + width * (len(pck_dists) - 1) / 2, all_scores)
                plt.legend()
                plt.grid(True, alpha=0.3, axis="y")

            # Overall title
            fig.suptitle(
                f"Ground Truth PCK-Brightness Analysis Dashboard\nDataset: {self.dataset_name}",
                fontsize=16,
            )

        except Exception as e:
            print(f"⚠️  Dashboard creation warning: {e}")

        # Save plot
        if save_plot:
            if filename is None:
                filename = f"gt_analysis_dashboard_{self.dataset_name}.svg"

            from config import ConfigManager

            config = ConfigManager.load_config(self.dataset_name)
            output_path = os.path.join(config.save_folder, filename)
            os.makedirs(config.save_folder, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches="tight", format="svg")
            plt.close()

            print(f"✅ GT analysis dashboard saved to: {output_path}")
            return output_path
        else:
            plt.show()
            return ""


def main():
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Ground Truth Visualization Creator")
    parser.add_argument("dataset", help="Dataset name (e.g., 'humaneva', 'movi')")
    parser.add_argument(
        "--joints", nargs="*", help="Specific joints to analyze (default: all)"
    )
    parser.add_argument("--pck-threshold", help="Specific PCK threshold to analyze")
    parser.add_argument(
        "--pck-scores", nargs="*", type=int, help="Specific PCK scores to plot"
    )
    parser.add_argument("--bin-size", type=int, default=5, help="Brightness bin size")
    parser.add_argument("--video", help="Path to specific video file")
    parser.add_argument("--subject", help="Filter ground truth by subject")
    parser.add_argument("--action", help="Filter ground truth by action")
    parser.add_argument("--camera", type=int, help="Filter ground truth by camera")
    parser.add_argument(
        "--plot-type",
        choices=[
            "pck_brightness",
            "joint_comparison",
            "correlation",
            "pck_distribution",
            "brightness_stats",
            "dashboard",
        ],
        default="dashboard",
        help="Type of plot to create",
    )
    parser.add_argument("--filename", help="Custom output filename")
    parser.add_argument(
        "--show", action="store_true", help="Show plot instead of saving"
    )

    args = parser.parse_args()

    try:
        creator = GTVisualizationCreator(args.dataset)

        # Set up ground truth filters
        gt_filters = {}
        if args.subject:
            gt_filters["subject"] = args.subject
        if args.action:
            gt_filters["action"] = args.action
        if args.camera is not None:
            gt_filters["camera"] = args.camera

        save_plot = not args.show

        if args.plot_type == "dashboard":
            # Run complete analysis for dashboard
            results = creator.calculator.run_complete_distribution_analysis(
                joint_names=args.joints,
                pck_threshold=args.pck_threshold,
                bin_size=args.bin_size,
                video_path=args.video,
                **gt_filters,
            )

            if results:
                creator.create_comprehensive_gt_analysis_dashboard(
                    results, save_plot, args.filename
                )

        else:
            # For other plot types, need to run analysis first
            results = creator.calculator.run_complete_distribution_analysis(
                joint_names=args.joints,
                pck_threshold=args.pck_threshold,
                bin_size=args.bin_size,
                video_path=args.video,
                **gt_filters,
            )

            if not results:
                print("❌ No analysis results available for plotting")
                return

            if args.plot_type == "pck_brightness":
                creator.create_pck_brightness_line_plot(
                    results["pck_brightness_distributions"],
                    args.pck_threshold,
                    args.pck_scores,
                    save_plot,
                    args.filename,
                )

            elif args.plot_type == "joint_comparison":
                creator.create_joint_brightness_comparison_plot(
                    results["joint_brightness_distributions"],
                    args.joints,
                    save_plot,
                    args.filename,
                )

            elif args.plot_type == "correlation":
                creator.create_correlation_heatmap(
                    results["analysis_results"], save_plot, args.filename
                )

            elif args.plot_type == "pck_distribution":
                creator.create_pck_score_distribution_plot(
                    results["pck_score_distributions"], save_plot, args.filename
                )

            elif args.plot_type == "brightness_stats":
                creator.create_brightness_statistics_plot(
                    results["joint_brightness_distributions"], save_plot, args.filename
                )

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

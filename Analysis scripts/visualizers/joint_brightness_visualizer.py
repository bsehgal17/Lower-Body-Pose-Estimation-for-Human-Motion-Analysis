"""
Joint Brightness Visualizer.

Creates visualizations for joint brightness analysis results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, Any
import os
from core.base_classes import BaseVisualizer


class JointBrightnessVisualizer(BaseVisualizer):
    """Visualizer for joint brightness analysis results."""

    def __init__(self, config):
        """Initialize joint brightness visualizer."""
        super().__init__(config)

        # Set style
        plt.style.use("seaborn-v0_8")
        sns.set_palette("husl")

    def create_plot(self, analysis_results: Dict[str, Any], save_path: str = None):
        """Create individual plots for each joint-PCK threshold combination."""
        if not analysis_results:
            print("❌ No analysis results to visualize")
            return

        print(f"Creating individual plots for {len(analysis_results)} joint metrics...")

        for metric_name, results in analysis_results.items():
            if not results or "pck_scores" not in results:
                continue

            self._create_individual_joint_plot(metric_name, results, save_path)

        print("✅ Individual plots created successfully")

    def _create_individual_joint_plot(
        self, metric_name: str, results: Dict[str, Any], save_path: str = None
    ):
        """Create plot for individual joint metric."""
        joint_name = results.get("joint_name", "Unknown")
        threshold = results.get("threshold", "0.05")

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(
            f"Joint Brightness Analysis: {joint_name} (PCK @ {threshold})",
            fontsize=16,
            fontweight="bold",
        )

        # 1. Scatter plot: PCK Score vs Brightness
        ax1 = axes[0, 0]
        pck_scores = results["pck_scores"]
        brightness_values = results["brightness_values"]

        ax1.scatter(pck_scores, brightness_values, alpha=0.6, s=30)
        ax1.set_xlabel("PCK Score")
        ax1.set_ylabel("Brightness Value")
        ax1.set_title("PCK Score vs Brightness")
        ax1.grid(True, alpha=0.3)

        # Add correlation info
        correlation = results.get("correlation", {}).get("pearson", 0.0)
        ax1.text(
            0.05,
            0.95,
            f"Correlation: {correlation:.3f}",
            transform=ax1.transAxes,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        # 2. Brightness distribution by PCK score ranges
        ax2 = axes[0, 1]
        score_ranges = results.get("score_ranges", {})

        if score_ranges:
            range_names = list(score_ranges.keys())
            mean_brightness = [
                score_ranges[name]["mean_brightness"] for name in range_names
            ]
            std_brightness = [
                score_ranges[name]["std_brightness"] for name in range_names
            ]

            bars = ax2.bar(
                range_names, mean_brightness, yerr=std_brightness, capsize=5, alpha=0.7
            )
            ax2.set_xlabel("PCK Score Range")
            ax2.set_ylabel("Mean Brightness")
            ax2.set_title("Brightness by PCK Score Range")

            # Add count labels on bars
            for i, (bar, range_name) in enumerate(zip(bars, range_names)):
                count = score_ranges[range_name]["count"]
                ax2.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + std_brightness[i] + 5,
                    f"n={count}",
                    ha="center",
                    va="bottom",
                )

        # 3. Brightness histogram
        ax3 = axes[1, 0]
        ax3.hist(brightness_values, bins=30, alpha=0.7, edgecolor="black")
        ax3.set_xlabel("Brightness Value")
        ax3.set_ylabel("Frequency")
        ax3.set_title("Brightness Distribution")
        ax3.axvline(
            np.mean(brightness_values),
            color="red",
            linestyle="--",
            label=f"Mean: {np.mean(brightness_values):.1f}",
        )
        ax3.legend()

        # 4. PCK score histogram
        ax4 = axes[1, 1]
        ax4.hist(pck_scores, bins=30, alpha=0.7, edgecolor="black")
        ax4.set_xlabel("PCK Score")
        ax4.set_ylabel("Frequency")
        ax4.set_title("PCK Score Distribution")
        ax4.axvline(
            np.mean(pck_scores),
            color="red",
            linestyle="--",
            label=f"Mean: {np.mean(pck_scores):.3f}",
        )
        ax4.legend()

        plt.tight_layout()

        # Save plot
        if save_path:
            filename = f"{save_path}_{joint_name}_{threshold.replace('.', '_')}.svg"
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            print(f"   Saved: {filename}")

        plt.close()

    def create_combined_summary_plot(
        self, analysis_results: Dict[str, Any], save_path: str = None
    ):
        """Create combined summary plot for all joints."""
        if not analysis_results:
            print("❌ No analysis results for summary plot")
            return

        print("Creating combined summary plot...")

        # Extract data for summary
        summary_data = []
        for metric_name, results in analysis_results.items():
            if not results or "joint_name" not in results:
                continue

            joint_name = results["joint_name"]
            threshold = results["threshold"]
            correlation = results.get("correlation", {}).get("pearson", 0.0)
            mean_brightness = (
                results.get("brightness_stats", {}).get("overall", {}).get("mean", 0.0)
            )
            total_frames = results.get("total_frames", 0)

            summary_data.append(
                {
                    "joint": joint_name,
                    "threshold": threshold,
                    "correlation": correlation,
                    "mean_brightness": mean_brightness,
                    "total_frames": total_frames,
                    "metric_name": metric_name,
                }
            )

        if not summary_data:
            print("❌ No valid data for summary plot")
            return

        df = pd.DataFrame(summary_data)

        # Create summary plot
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(
            "Joint Brightness Analysis Summary", fontsize=16, fontweight="bold"
        )

        # 1. Correlation by joint and threshold
        ax1 = axes[0, 0]
        pivot_corr = df.pivot(index="joint", columns="threshold", values="correlation")
        sns.heatmap(pivot_corr, annot=True, cmap="RdBu_r", center=0, ax=ax1)
        ax1.set_title("PCK-Brightness Correlation by Joint")
        ax1.set_xlabel("PCK Threshold")
        ax1.set_ylabel("Joint")

        # 2. Mean brightness by joint
        ax2 = axes[0, 1]
        mean_brightness_by_joint = (
            df.groupby("joint")["mean_brightness"].mean().sort_values(ascending=False)
        )
        bars = ax2.bar(
            range(len(mean_brightness_by_joint)), mean_brightness_by_joint.values
        )
        ax2.set_xticks(range(len(mean_brightness_by_joint)))
        ax2.set_xticklabels(mean_brightness_by_joint.index, rotation=45, ha="right")
        ax2.set_ylabel("Mean Brightness")
        ax2.set_title("Average Brightness by Joint")

        # Add value labels on bars
        for i, bar in enumerate(bars):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{mean_brightness_by_joint.values[i]:.1f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        # 3. Correlation distribution
        ax3 = axes[1, 0]
        ax3.hist(df["correlation"], bins=20, alpha=0.7, edgecolor="black")
        ax3.set_xlabel("PCK-Brightness Correlation")
        ax3.set_ylabel("Frequency")
        ax3.set_title("Distribution of Correlations")
        ax3.axvline(
            df["correlation"].mean(),
            color="red",
            linestyle="--",
            label=f"Mean: {df['correlation'].mean():.3f}",
        )
        ax3.legend()

        # 4. Frame count by joint
        ax4 = axes[1, 1]
        frames_by_joint = (
            df.groupby("joint")["total_frames"].mean().sort_values(ascending=False)
        )
        ax4.bar(range(len(frames_by_joint)), frames_by_joint.values)
        ax4.set_xticks(range(len(frames_by_joint)))
        ax4.set_xticklabels(frames_by_joint.index, rotation=45, ha="right")
        ax4.set_ylabel("Number of Frames")
        ax4.set_title("Analysis Frame Count by Joint")

        plt.tight_layout()

        # Save plot
        if save_path:
            filename = f"{save_path}_summary.svg"
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            print(f"   Saved: {filename}")

        plt.close()
        print("✅ Combined summary plot created successfully")

    def create_joint_comparison_plot(
        self, analysis_results: Dict[str, Any], save_path: str = None
    ):
        """Create comparison plot between different joints."""
        if not analysis_results:
            print("❌ No analysis results for comparison plot")
            return

        print("Creating joint comparison plot...")

        # Group results by threshold
        threshold_groups = {}
        for metric_name, results in analysis_results.items():
            if not results or "threshold" not in results:
                continue

            threshold = results["threshold"]
            if threshold not in threshold_groups:
                threshold_groups[threshold] = {}

            joint_name = results["joint_name"]
            threshold_groups[threshold][joint_name] = results

        # Create plots for each threshold
        for threshold, joint_results in threshold_groups.items():
            if len(joint_results) < 2:
                continue  # Need at least 2 joints for comparison

            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle(
                f"Joint Comparison (PCK Threshold: {threshold})",
                fontsize=16,
                fontweight="bold",
            )

            # Prepare data
            joint_names = list(joint_results.keys())
            correlations = [
                joint_results[joint]["correlation"]["pearson"] for joint in joint_names
            ]
            mean_brightness = [
                joint_results[joint]["brightness_stats"]["overall"]["mean"]
                for joint in joint_names
            ]
            frame_counts = [
                joint_results[joint]["total_frames"] for joint in joint_names
            ]

            # 1. Correlation comparison
            ax1 = axes[0]
            bars1 = ax1.bar(joint_names, correlations)
            ax1.set_ylabel("PCK-Brightness Correlation")
            ax1.set_title("Correlation Comparison")
            ax1.tick_params(axis="x", rotation=45)

            # Color bars based on correlation strength
            for bar, corr in zip(bars1, correlations):
                if abs(corr) > 0.5:
                    bar.set_color("darkgreen" if corr > 0 else "darkred")
                elif abs(corr) > 0.3:
                    bar.set_color("orange")
                else:
                    bar.set_color("gray")

            # 2. Mean brightness comparison
            ax2 = axes[1]
            ax2.bar(joint_names, mean_brightness)
            ax2.set_ylabel("Mean Brightness")
            ax2.set_title("Average Brightness Comparison")
            ax2.tick_params(axis="x", rotation=45)

            # 3. Sample size comparison
            ax3 = axes[2]
            ax3.bar(joint_names, frame_counts)
            ax3.set_ylabel("Number of Frames")
            ax3.set_title("Sample Size Comparison")
            ax3.tick_params(axis="x", rotation=45)

            plt.tight_layout()

            # Save plot
            if save_path:
                filename = f"{save_path}_comparison_{threshold.replace('.', '_')}.svg"
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                plt.savefig(filename, dpi=300, bbox_inches="tight")
                print(f"   Saved: {filename}")

            plt.close()

        print("✅ Joint comparison plots created successfully")

    def create_brightness_heatmap(
        self, analysis_results: Dict[str, Any], save_path: str = None
    ):
        """Create heatmap showing brightness patterns across joints and thresholds."""
        if not analysis_results:
            print("❌ No analysis results for heatmap")
            return

        print("Creating brightness heatmap...")

        # Prepare data for heatmap
        data_for_heatmap = []
        for metric_name, results in analysis_results.items():
            if not results or "joint_name" not in results:
                continue

            joint_name = results["joint_name"]
            threshold = results["threshold"]
            mean_brightness = (
                results.get("brightness_stats", {}).get("overall", {}).get("mean", 0.0)
            )

            data_for_heatmap.append(
                {
                    "joint": joint_name,
                    "threshold": threshold,
                    "mean_brightness": mean_brightness,
                }
            )

        if not data_for_heatmap:
            print("❌ No valid data for heatmap")
            return

        df = pd.DataFrame(data_for_heatmap)
        pivot_df = df.pivot(
            index="joint", columns="threshold", values="mean_brightness"
        )

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            pivot_df,
            annot=True,
            cmap="viridis",
            fmt=".1f",
            cbar_kws={"label": "Mean Brightness"},
        )
        plt.title(
            "Joint Brightness Heatmap by PCK Threshold", fontsize=14, fontweight="bold"
        )
        plt.xlabel("PCK Threshold")
        plt.ylabel("Joint")
        plt.tight_layout()

        # Save plot
        if save_path:
            filename = f"{save_path}_heatmap.svg"
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            print(f"   Saved: {filename}")

        plt.close()
        print("✅ Brightness heatmap created successfully")

    def create_per_frame_scatter_plots(
        self, analysis_results: Dict[str, Any], save_path: str = None
    ):
        """Create per-frame scatter plots for each joint showing PCK vs brightness over time."""
        if not analysis_results:
            print("❌ No analysis results for per-frame scatter plots")
            return

        print("Creating per-frame scatter plots...")

        for metric_name, results in analysis_results.items():
            if not results or "pck_scores" not in results:
                continue

            joint_name = results["joint_name"]
            threshold = results["threshold"]
            pck_scores = results["pck_scores"]
            brightness_values = results["brightness_values"]

            if len(pck_scores) != len(brightness_values):
                print(f"   Skipping {joint_name} - mismatched data lengths")
                continue

            # Create scatter plot
            plt.figure(figsize=(12, 8))

            # Color points by frame number for temporal visualization
            frame_numbers = range(len(pck_scores))
            scatter = plt.scatter(
                brightness_values,
                pck_scores,
                c=frame_numbers,
                cmap="viridis",
                alpha=0.6,
                s=20,
            )

            # Add colorbar for frame numbers
            cbar = plt.colorbar(scatter)
            cbar.set_label("Frame Number", rotation=270, labelpad=15)

            # Add trend line
            if len(pck_scores) > 1:
                z = np.polyfit(brightness_values, pck_scores, 1)
                p = np.poly1d(z)
                x_trend = np.linspace(
                    min(brightness_values), max(brightness_values), 100
                )
                plt.plot(
                    x_trend,
                    p(x_trend),
                    "r--",
                    alpha=0.8,
                    linewidth=2,
                    label="Trend Line",
                )

            # Calculate correlation
            correlation = np.corrcoef(brightness_values, pck_scores)[0, 1]

            plt.xlabel("Brightness Value")
            plt.ylabel("PCK Score")
            plt.title(
                f"Per-Frame: {joint_name} PCK vs Brightness (Threshold: {threshold})\n"
                f"Correlation: {correlation:.3f}"
            )
            plt.grid(True, alpha=0.3)
            plt.legend()

            # Save plot
            if save_path:
                filename = f"{save_path}_per_frame_{joint_name}_{threshold.replace('.', '_')}.svg"
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                plt.savefig(filename, dpi=300, bbox_inches="tight")
                print(f"   Saved: {filename}")

            plt.close()

        print("✅ Per-frame scatter plots created successfully")

    def create_per_frame_line_plots(
        self, analysis_results: Dict[str, Any], save_path: str = None
    ):
        """Create per-frame line plots showing PCK and brightness evolution over time."""
        if not analysis_results:
            print("❌ No analysis results for per-frame line plots")
            return

        print("Creating per-frame line plots...")

        for metric_name, results in analysis_results.items():
            if not results or "pck_scores" not in results:
                continue

            joint_name = results["joint_name"]
            threshold = results["threshold"]
            pck_scores = results["pck_scores"]
            brightness_values = results["brightness_values"]

            if len(pck_scores) != len(brightness_values):
                print(f"   Skipping {joint_name} - mismatched data lengths")
                continue

            # Create line plot with dual y-axis
            fig, ax1 = plt.subplots(figsize=(14, 8))

            frame_numbers = range(len(pck_scores))

            # Plot PCK scores
            color1 = "tab:red"
            ax1.set_xlabel("Frame Number")
            ax1.set_ylabel("PCK Score", color=color1)
            line1 = ax1.plot(
                frame_numbers,
                pck_scores,
                color=color1,
                linewidth=2,
                label=f"PCK Score (threshold: {threshold})",
                alpha=0.8,
            )
            ax1.tick_params(axis="y", labelcolor=color1)
            ax1.grid(True, alpha=0.3)

            # Create second y-axis for brightness
            ax2 = ax1.twinx()
            color2 = "tab:blue"
            ax2.set_ylabel("Brightness Value", color=color2)
            line2 = ax2.plot(
                frame_numbers,
                brightness_values,
                color=color2,
                linewidth=2,
                label="Brightness",
                alpha=0.8,
            )
            ax2.tick_params(axis="y", labelcolor=color2)

            # Add title and legend
            plt.title(f"Per-Frame Evolution: {joint_name} (Threshold: {threshold})")

            # Combine legends from both axes
            lines = line1 + line2
            labels = [line.get_label() for line in lines]
            ax1.legend(lines, labels, loc="upper left")

            # Add correlation info
            correlation = np.corrcoef(brightness_values, pck_scores)[0, 1]
            ax1.text(
                0.02,
                0.98,
                f"Correlation: {correlation:.3f}",
                transform=ax1.transAxes,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                verticalalignment="top",
            )

            plt.tight_layout()

            # Save plot
            if save_path:
                filename = (
                    f"{save_path}_line_{joint_name}_{threshold.replace('.', '_')}.svg"
                )
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                plt.savefig(filename, dpi=300, bbox_inches="tight")
                print(f"   Saved: {filename}")

            plt.close()

        print("✅ Per-frame line plots created successfully")

    def create_joint_correlation_over_time(
        self,
        analysis_results: Dict[str, Any],
        save_path: str = None,
        window_size: int = 50,
    ):
        """Create plots showing how correlation changes over time using rolling windows."""
        if not analysis_results:
            print("❌ No analysis results for correlation over time plots")
            return

        print("Creating correlation over time plots...")

        # Group results by joint for multi-threshold analysis
        joint_groups = {}
        for metric_name, results in analysis_results.items():
            if not results or "joint_name" not in results:
                continue

            joint_name = results["joint_name"]
            if joint_name not in joint_groups:
                joint_groups[joint_name] = {}

            threshold = results["threshold"]
            joint_groups[joint_name][threshold] = results

        # Create correlation over time plot for each joint
        for joint_name, thresholds in joint_groups.items():
            plt.figure(figsize=(14, 8))

            colors = plt.cm.viridis(np.linspace(0, 1, len(thresholds)))

            for i, (threshold, results) in enumerate(thresholds.items()):
                pck_scores = results["pck_scores"]
                brightness_values = results["brightness_values"]

                if len(pck_scores) < window_size:
                    print(
                        f"   Skipping {joint_name} threshold {threshold} - insufficient data"
                    )
                    continue

                # Calculate rolling correlation
                correlations = []
                frame_centers = []

                for start_idx in range(len(pck_scores) - window_size + 1):
                    end_idx = start_idx + window_size
                    window_pck = pck_scores[start_idx:end_idx]
                    window_brightness = brightness_values[start_idx:end_idx]

                    if len(set(window_pck)) > 1 and len(set(window_brightness)) > 1:
                        corr = np.corrcoef(window_brightness, window_pck)[0, 1]
                        correlations.append(corr)
                        frame_centers.append(start_idx + window_size // 2)

                if correlations:
                    plt.plot(
                        frame_centers,
                        correlations,
                        color=colors[i],
                        linewidth=2,
                        label=f"Threshold {threshold}",
                        alpha=0.8,
                    )

            plt.xlabel("Frame Number")
            plt.ylabel("Correlation Coefficient")
            plt.title(
                f"Rolling Correlation Over Time: {joint_name}\n"
                f"(Window Size: {window_size} frames)"
            )
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.axhline(y=0, color="black", linestyle="--", alpha=0.5)

            # Save plot
            if save_path:
                filename = (
                    f"{save_path}_correlation_time_{joint_name.replace(' ', '_')}.svg"
                )
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                plt.savefig(filename, dpi=300, bbox_inches="tight")
                print(f"   Saved: {filename}")

            plt.close()

        print("✅ Correlation over time plots created successfully")

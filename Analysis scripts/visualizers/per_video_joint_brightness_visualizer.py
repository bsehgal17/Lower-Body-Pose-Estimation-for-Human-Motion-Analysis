"""
Per-Video Joint Brightness Visualizer

Creates simplified visualization for per-video joint brightness analysis results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional
import os
from core.base_classes import BaseVisualizer


class PerVideoJointBrightnessVisualizer(BaseVisualizer):
    """Simplified visualizer for per-video joint brightness analysis results."""

    def __init__(self, output_dir: str = None, save_plots: bool = True):
        """Initialize the per-video visualizer.

        Args:
            output_dir: Directory to save plots (optional)
            save_plots: Whether to save plots to files
        """
        super().__init__(config=None)  # Pass None as config to satisfy base class
        self.output_dir = output_dir
        self.save_plots = save_plots

        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)

    def create_plot(self, data: pd.DataFrame, **kwargs) -> None:
        """Create a plot from the given data (required by BaseVisualizer).

        This method delegates to create_all_visualizations for compatibility.
        """
        # Convert DataFrame to analysis_results format if needed
        if isinstance(data, dict):
            analysis_results = data
        else:
            # For compatibility, if called with DataFrame, create minimal structure
            analysis_results = {"default_video": {"brightness_summary": {}}}

        self.create_all_visualizations(analysis_results)

    def create_all_visualizations(self, analysis_results: Dict[str, Any]) -> None:
        """Create simplified visualization showing average PCK vs brightness per video for joints.

        Args:
            analysis_results: Results from PerVideoJointBrightnessAnalyzer
        """
        print("Creating per-video joint brightness visualization...")

        if not analysis_results:
            print("❌ No analysis results to visualize")
            return

        # Create single plot: average PCK vs brightness per video for joints
        self.create_pck_brightness_plot(analysis_results)

        print("✅ Per-video visualization completed")

    def create_pck_brightness_plot(self, analysis_results: Dict[str, Any]) -> None:
        """Create single plot showing average PCK vs brightness per video for joints."""

        print("Creating PCK vs brightness plot...")

        # Prepare data for plotting
        plot_data = []

        for video_name, video_results in analysis_results.items():
            # Get brightness summary for this video
            brightness_summary = video_results.get("brightness_summary", {})

            # Get PCK data for this video and calculate averages per joint
            pck_by_joint = {}

            for pck_column, pck_results in video_results.items():
                if pck_column in [
                    "video_name",
                    "total_frames",
                    "joints_analyzed",
                    "brightness_summary",
                ]:
                    continue

                joint_name = pck_results.get("joint_name", "unknown")
                pck_scores = pck_results.get("pck_scores", [])

                if pck_scores and joint_name != "unknown":
                    if joint_name not in pck_by_joint:
                        pck_by_joint[joint_name] = []
                    pck_by_joint[joint_name].extend(pck_scores)

            # Create data points for each joint in this video
            for joint_name in brightness_summary.keys():
                brightness_mean = brightness_summary[joint_name]["mean"]

                if joint_name in pck_by_joint and brightness_mean > 0:
                    pck_mean = np.mean(pck_by_joint[joint_name])

                    plot_data.append(
                        {
                            "video": video_name,
                            "joint": joint_name,
                            "avg_brightness": brightness_mean,
                            "avg_pck": pck_mean,
                        }
                    )

        if not plot_data:
            print("❌ No PCK-brightness data found for plotting")
            return

        df = pd.DataFrame(plot_data)

        # Create the plot
        plt.figure(figsize=(12, 8))

        # Create scatter plot with different colors for each joint
        joints = df["joint"].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(joints)))

        for i, joint in enumerate(joints):
            joint_data = df[df["joint"] == joint]
            plt.scatter(
                joint_data["avg_brightness"],
                joint_data["avg_pck"],
                label=joint,
                color=colors[i],
                alpha=0.7,
                s=100,
            )

            # Add video labels for each point
            for _, row in joint_data.iterrows():
                plt.annotate(
                    row["video"],
                    (row["avg_brightness"], row["avg_pck"]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                    alpha=0.8,
                )

        plt.xlabel("Average Brightness", fontsize=12)
        plt.ylabel("Average PCK Score", fontsize=12)
        plt.title(
            "Average PCK vs Brightness per Video for Joints",
            fontsize=14,
            fontweight="bold",
        )
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if self.save_plots and self.output_dir:
            filename = os.path.join(self.output_dir, "pck_vs_brightness_plot.png")
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            print(f"   Saved: {filename}")

        plt.show()

    def save_results_to_csv(self, analysis_results: Dict[str, Any]) -> None:
        """Save analysis results to CSV files for further analysis."""

        if not self.output_dir:
            print("❌ No output directory specified for CSV export")
            return

        print("Saving results to CSV files...")

        # 1. Video summary CSV
        video_summaries = []
        for video_name, video_results in analysis_results.items():
            brightness_summary = video_results.get("brightness_summary", {})

            video_summary = {
                "video_name": video_name,
                "total_frames": video_results.get("total_frames", 0),
                "joints_analyzed": len(video_results.get("joints_analyzed", [])),
            }

            # Add brightness metrics for each joint
            for joint_name, stats in brightness_summary.items():
                video_summary[f"{joint_name}_mean_brightness"] = stats["mean"]
                video_summary[f"{joint_name}_std_brightness"] = stats["std"]
                video_summary[f"{joint_name}_valid_frames"] = stats["valid_frames"]

            video_summaries.append(video_summary)

        video_df = pd.DataFrame(video_summaries)
        video_csv_path = os.path.join(self.output_dir, "video_brightness_summary.csv")
        video_df.to_csv(video_csv_path, index=False)
        print(f"   Saved video summary: {video_csv_path}")

        # 2. Detailed results CSV
        detailed_results = []
        for video_name, video_results in analysis_results.items():
            for pck_column, pck_results in video_results.items():
                if pck_column in [
                    "video_name",
                    "total_frames",
                    "joints_analyzed",
                    "brightness_summary",
                ]:
                    continue

                result_row = {
                    "video_name": video_name,
                    "pck_column": pck_column,
                    "joint_name": pck_results.get("joint_name", "unknown"),
                    "threshold": pck_results.get("threshold", "unknown"),
                    "valid_frames": pck_results.get("valid_frames", 0),
                    "pearson_correlation": pck_results.get("correlation", {}).get(
                        "pearson", 0.0
                    ),
                    "spearman_correlation": pck_results.get("correlation", {}).get(
                        "spearman", 0.0
                    ),
                }

                # Add brightness and PCK statistics
                brightness_stats = pck_results.get("brightness_stats", {})
                pck_stats = pck_results.get("pck_stats", {})

                for stat_name, stat_value in brightness_stats.items():
                    result_row[f"brightness_{stat_name}"] = stat_value

                for stat_name, stat_value in pck_stats.items():
                    result_row[f"pck_{stat_name}"] = stat_value

                detailed_results.append(result_row)

        detailed_df = pd.DataFrame(detailed_results)
        detailed_csv_path = os.path.join(
            self.output_dir, "detailed_pck_brightness_results.csv"
        )
        detailed_df.to_csv(detailed_csv_path, index=False)
        print(f"   Saved detailed results: {detailed_csv_path}")

        print("✅ CSV export completed")

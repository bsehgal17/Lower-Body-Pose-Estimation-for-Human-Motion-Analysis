"""
Per-Video Joint Brightness Visualizer

Creates simplified visualization for per-video joint brightness analysis results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any
import os
from core.base_classes import BaseVisualizer


class PerVideoJointBrightnessVisualizer(BaseVisualizer):
    """Simplified visualizer for per-video joint brightness analysis results."""

    def __init__(
        self,
        output_dir: str = None,
        save_plots: bool = True,
        create_individual_plots: bool = False,
    ):
        """Initialize the per-video visualizer.

        Args:
            output_dir: Directory to save plots (optional)
            save_plots: Whether to save plots to files
            create_individual_plots: Whether to create individual plots for each video (default: False)
        """
        super().__init__(config=None)  # Pass None as config to satisfy base class
        self.output_dir = output_dir
        self.save_plots = save_plots
        self.create_individual_plots = create_individual_plots

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
        """Create comprehensive visualizations for per-video joint brightness analysis.

        Args:
            analysis_results: Results from PerVideoJointBrightnessAnalyzer
        """
        print("Creating per-video joint brightness visualizations...")

        if not analysis_results:
            print("❌ No analysis results to visualize")
            return

        # Create combined scatter plot showing all videos and joints together (MAIN PLOT)
        self.create_combined_scatter_plot(analysis_results)

        # Create aggregated plot: average PCK vs brightness per video for joints
        self.create_pck_brightness_plot(analysis_results)

        # Create individual scatter plots for each video (optional)
        if self.create_individual_plots:
            self.create_per_video_scatter_plots(analysis_results)

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

    def create_per_video_scatter_plots(self, analysis_results: Dict[str, Any]) -> None:
        """Create individual scatter plots for each video showing frame-by-frame PCK vs brightness for all joints.

        This creates the scatter plot you requested: PCK vs brightness per video where brightness
        is taken around joint from ground truth coordinates.
        """
        print(
            "Creating per-video scatter plots (PCK vs brightness from GT joint locations)..."
        )

        for video_name, video_results in analysis_results.items():
            print(f"   Creating scatter plot for video: {video_name}")

            # Collect frame-by-frame data for all joints in this video
            plot_data = []

            for pck_column, pck_results in video_results.items():
                if pck_column in [
                    "video_name",
                    "total_frames",
                    "joints_analyzed",
                    "brightness_summary",
                ]:
                    continue

                joint_name = pck_results.get("joint_name", "unknown")
                threshold = pck_results.get("threshold", "unknown")

                # Get frame-by-frame PCK scores and brightness values
                pck_scores = pck_results.get("pck_scores", [])
                brightness_values = pck_results.get("brightness_values", [])

                if not pck_scores or not brightness_values:
                    continue

                # Add data points for this joint
                for pck, brightness in zip(pck_scores, brightness_values):
                    plot_data.append(
                        {
                            "joint": joint_name,
                            "threshold": threshold,
                            "pck": pck,
                            "brightness": brightness,
                            "pck_column": pck_column,
                        }
                    )

            if not plot_data:
                print(f"   ❌ No frame-by-frame data found for {video_name}")
                continue

            # Create the scatter plot for this video
            df = pd.DataFrame(plot_data)

            # Group by joint for consistent coloring across thresholds
            unique_joints = df["joint"].unique()

            # Create subplots if multiple thresholds exist
            unique_thresholds = sorted(df["threshold"].unique())
            n_thresholds = len(unique_thresholds)

            if n_thresholds == 1:
                fig, ax = plt.subplots(figsize=(12, 8))
                axes = [ax]
            else:
                fig, axes = plt.subplots(1, n_thresholds, figsize=(6 * n_thresholds, 8))
                if n_thresholds == 1:
                    axes = [axes]

            # Color palette for joints
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_joints)))
            joint_colors = dict(zip(unique_joints, colors))

            # Plot each threshold in its own subplot
            for idx, threshold in enumerate(unique_thresholds):
                ax = axes[idx]
                threshold_data = df[df["threshold"] == threshold]

                # Plot each joint with its unique color
                for joint in unique_joints:
                    joint_data = threshold_data[threshold_data["joint"] == joint]

                    if len(joint_data) > 0:
                        ax.scatter(
                            joint_data["brightness"],
                            joint_data["pck"],
                            label=joint,
                            color=joint_colors[joint],
                            alpha=0.6,
                            s=20,
                        )

                # Calculate correlation for this threshold
                if len(threshold_data) > 1:
                    correlation = np.corrcoef(
                        threshold_data["brightness"], threshold_data["pck"]
                    )[0, 1]
                    if not np.isnan(correlation):
                        ax.text(
                            0.05,
                            0.95,
                            f"r = {correlation:.3f}",
                            transform=ax.transAxes,
                            fontsize=12,
                            bbox=dict(
                                boxstyle="round,pad=0.3", facecolor="white", alpha=0.8
                            ),
                        )

                ax.set_xlabel("Brightness (from GT Joint Location)", fontsize=12)
                ax.set_ylabel("PCK Score", fontsize=12)
                ax.set_title(f"PCK vs Brightness - Threshold {threshold}", fontsize=14)
                ax.grid(True, alpha=0.3)
                ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

            # Overall figure title
            plt.suptitle(
                f"Frame-by-Frame PCK vs Brightness Analysis\nVideo: {video_name}",
                fontsize=16,
                fontweight="bold",
                y=0.98,
            )
            plt.tight_layout()

            # Save the plot
            if self.save_plots and self.output_dir:
                # Clean video name for filename
                clean_video_name = (
                    str(video_name)
                    .replace("/", "_")
                    .replace("\\", "_")
                    .replace(":", "_")
                )
                filename = os.path.join(
                    self.output_dir, f"pck_brightness_scatter_{clean_video_name}.png"
                )
                plt.savefig(filename, dpi=300, bbox_inches="tight")
                print(f"   Saved: {filename}")

            plt.show()
            plt.close()

        print("✅ Per-video scatter plots completed")

    def create_combined_scatter_plot(self, analysis_results: Dict[str, Any]) -> None:
        """Create a combined scatter plot showing PCK vs brightness for all videos and joints together.

        This creates a single plot with all frame-by-frame data points from all videos.
        """
        print("Creating combined scatter plot (all videos and joints together)...")

        # Ensure matplotlib is in interactive mode for plot display
        plt.ion()

        # Collect all frame-by-frame data from all videos
        all_data = []

        for video_name, video_results in analysis_results.items():
            for pck_column, pck_results in video_results.items():
                if pck_column in [
                    "video_name",
                    "total_frames",
                    "joints_analyzed",
                    "brightness_summary",
                ]:
                    continue

                joint_name = pck_results.get("joint_name", "unknown")
                threshold = pck_results.get("threshold", "unknown")

                # Get frame-by-frame PCK scores and brightness values
                pck_scores = pck_results.get("pck_scores", [])
                brightness_values = pck_results.get("brightness_values", [])

                if not pck_scores or not brightness_values:
                    continue

                # Add data points for this joint/video combination
                for pck, brightness in zip(pck_scores, brightness_values):
                    all_data.append(
                        {
                            "video": str(video_name),
                            "joint": joint_name,
                            "threshold": threshold,
                            "pck": pck,
                            "brightness": brightness,
                        }
                    )

        if not all_data:
            print("❌ No combined data found for scatter plot")
            return

        df = pd.DataFrame(all_data)
        print(f"   Total data points: {len(df)}")
        print(f"   Videos: {df['video'].nunique()}")
        print(f"   Joints: {df['joint'].nunique()}")
        print(f"   Thresholds: {df['threshold'].nunique()}")

        # Create subplots for different thresholds if multiple exist
        unique_thresholds = sorted(df["threshold"].unique())
        n_thresholds = len(unique_thresholds)

        if n_thresholds == 1:
            fig, ax = plt.subplots(figsize=(14, 10))
            axes = [ax]
        else:
            fig, axes = plt.subplots(1, n_thresholds, figsize=(7 * n_thresholds, 10))
            if n_thresholds == 1:
                axes = [axes]

        # Color palette for joints
        unique_joints = df["joint"].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_joints)))
        joint_colors = dict(zip(unique_joints, colors))

        # Plot each threshold
        for idx, threshold in enumerate(unique_thresholds):
            ax = axes[idx]
            threshold_data = df[df["threshold"] == threshold]

            # Plot each joint with its unique color
            for joint in unique_joints:
                joint_data = threshold_data[threshold_data["joint"] == joint]

                if len(joint_data) > 0:
                    ax.scatter(
                        joint_data["brightness"],
                        joint_data["pck"],
                        label=joint,
                        color=joint_colors[joint],
                        alpha=0.5,
                        s=15,
                        edgecolors="black",
                        linewidth=0.3,
                    )

            # Calculate and display overall correlation for this threshold
            if len(threshold_data) > 1:
                correlation = np.corrcoef(
                    threshold_data["brightness"], threshold_data["pck"]
                )[0, 1]
                if not np.isnan(correlation):
                    ax.text(
                        0.05,
                        0.95,
                        f"Overall r = {correlation:.3f}",
                        transform=ax.transAxes,
                        fontsize=14,
                        fontweight="bold",
                        bbox=dict(
                            boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.8
                        ),
                    )

            # Add trend line
            if len(threshold_data) > 10:  # Only if we have enough points
                z = np.polyfit(threshold_data["brightness"], threshold_data["pck"], 1)
                p = np.poly1d(z)
                x_trend = np.linspace(
                    threshold_data["brightness"].min(),
                    threshold_data["brightness"].max(),
                    100,
                )
                ax.plot(
                    x_trend,
                    p(x_trend),
                    "r--",
                    alpha=0.8,
                    linewidth=2,
                    label="Trend line",
                )

            ax.set_xlabel("Brightness (from GT Joint Location)", fontsize=14)
            ax.set_ylabel("PCK Score", fontsize=14)
            ax.set_title(
                f"All Videos - PCK vs Brightness\nThreshold: {threshold}",
                fontsize=16,
                fontweight="bold",
            )
            ax.grid(True, alpha=0.3)
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

            # Set reasonable axis limits
            ax.set_xlim(
                threshold_data["brightness"].min() * 0.95,
                threshold_data["brightness"].max() * 1.05,
            )
            ax.set_ylim(-0.05, 1.05)

        # Overall figure title
        plt.suptitle(
            f"Combined PCK vs Brightness Analysis\nAll Videos and Joints ({len(df)} data points)",
            fontsize=18,
            fontweight="bold",
            y=0.98,
        )
        plt.tight_layout()

        # Save the combined plot
        if self.save_plots and self.output_dir:
            filename = os.path.join(
                self.output_dir, "combined_pck_brightness_scatter.png"
            )
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            print(f"   Saved combined plot: {filename}")

        plt.show()
        plt.close()

        print("✅ Combined scatter plot completed")

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

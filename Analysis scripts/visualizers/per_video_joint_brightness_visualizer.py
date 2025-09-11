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

        # Debug: Print analysis results structure
        print(f"   Debug: Received analysis results for {len(analysis_results)} videos")
        for i, (video_name, video_data) in enumerate(analysis_results.items()):
            if i < 2:  # Show first 2 videos
                print(f"   Video '{video_name}' structure:")
                for key, value in video_data.items():
                    if isinstance(value, dict) and key not in ["brightness_summary"]:
                        has_pck_scores = "pck_scores" in value
                        has_brightness = "brightness_values" in value
                        pck_len = (
                            len(value.get("pck_scores", [])) if has_pck_scores else 0
                        )
                        brightness_len = (
                            len(value.get("brightness_values", []))
                            if has_brightness
                            else 0
                        )
                        print(
                            f"      {key}: pck_scores={pck_len}, brightness_values={brightness_len}"
                        )
                    else:
                        print(f"      {key}: {type(value)}")
            elif i == 2:
                print("   ... (more videos)")
                break

        # Create the main requested plot: average PCK vs brightness per video (color-coded by video)
        self.create_pck_brightness_plot(analysis_results)

        # Create combined scatter plot showing all videos and joints together (detailed plot)
        if self.create_individual_plots:
            self.create_combined_scatter_plot(analysis_results)

        # Create individual scatter plots for each video (optional)
        if self.create_individual_plots:
            self.create_per_video_scatter_plots(analysis_results)

        print("✅ Per-video visualization completed")

    def create_pck_brightness_plot(self, analysis_results: Dict[str, Any]) -> None:
        """Create single plot showing average PCK vs brightness per video, color-coded by video."""

        print("Creating combined average PCK vs brightness plot per video...")

        # Prepare data for plotting - combine all joints per video
        plot_data = []

        for video_name, video_results in analysis_results.items():
            # Get brightness summary for this video
            brightness_summary = video_results.get("brightness_summary", {})

            # Get PCK data for this video and calculate averages across all joints
            all_pck_scores = []
            all_brightness_values = []

            # Collect all PCK scores and brightness values for all joints in this video
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

                if (
                    pck_scores
                    and joint_name != "unknown"
                    and joint_name in brightness_summary
                ):
                    # Add all PCK scores for this joint
                    all_pck_scores.extend(pck_scores)

                    # Add corresponding brightness values
                    brightness_values = pck_results.get("brightness_values", [])
                    if brightness_values:
                        all_brightness_values.extend(brightness_values)

            # Also include average brightness across all joints
            if brightness_summary:
                joint_brightness_averages = [
                    stats["mean"]
                    for stats in brightness_summary.values()
                    if stats["mean"] > 0
                ]
                if joint_brightness_averages:
                    avg_brightness_all_joints = np.mean(joint_brightness_averages)
                else:
                    avg_brightness_all_joints = 0

            # Calculate overall averages for this video
            if all_pck_scores and all_brightness_values:
                avg_pck = np.mean(all_pck_scores)
                avg_brightness_from_frames = np.mean(all_brightness_values)

                plot_data.append(
                    {
                        "video": str(video_name),
                        "avg_pck": avg_pck,
                        "avg_brightness": avg_brightness_from_frames,
                        "avg_brightness_joints": avg_brightness_all_joints,
                        "total_data_points": len(all_pck_scores),
                        "num_joints": len(brightness_summary),
                    }
                )

        if not plot_data:
            print("❌ No PCK-brightness data found for plotting")
            return

        df = pd.DataFrame(plot_data)
        print(f"   Created plot data for {len(df)} videos")

        # Create the plot
        plt.figure(figsize=(14, 10))

        # Create scatter plot with different colors for each video
        videos = df["video"].unique()
        colors = plt.cm.Set3(np.linspace(0, 1, len(videos)))

        for i, video in enumerate(videos):
            video_data = df[df["video"] == video].iloc[0]

            plt.scatter(
                video_data["avg_brightness"],
                video_data["avg_pck"],
                label=video,
                color=colors[i],
                alpha=0.8,
                s=150,
                edgecolors="black",
                linewidth=1,
            )

        # Add trend line if we have enough data points
        if len(df) > 2:
            z = np.polyfit(df["avg_brightness"], df["avg_pck"], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(
                df["avg_brightness"].min(), df["avg_brightness"].max(), 100
            )
            plt.plot(
                x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2, label="Trend line"
            )

        plt.xlabel(
            "Average Brightness (Combined All Joints)", fontsize=14, fontweight="bold"
        )
        plt.ylabel(
            "Average PCK Score (Combined All Joints)", fontsize=14, fontweight="bold"
        )
        plt.title(
            f"Average PCK vs Brightness Per Video\n(Combined across all 6 joints: {len(df)} videos)",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )

        # Customize the legend with multiple columns (10 videos per column)
        num_videos = len(videos)
        num_cols = max(1, (num_videos + 9) // 10)  # Calculate number of columns needed
        plt.legend(
            bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10, ncol=num_cols
        )
        plt.grid(True, alpha=0.3)

        # Set reasonable axis limits with some padding
        x_range = df["avg_brightness"].max() - df["avg_brightness"].min()
        y_range = df["avg_pck"].max() - df["avg_pck"].min()

        plt.xlim(
            df["avg_brightness"].min() - 0.05 * x_range,
            df["avg_brightness"].max() + 0.05 * x_range,
        )
        plt.ylim(
            max(0, df["avg_pck"].min() - 0.05 * y_range),
            min(1, df["avg_pck"].max() + 0.05 * y_range),
        )

        plt.tight_layout()

        # Add summary statistics as text
        stats_text = "Statistics:\n"
        stats_text += f"Videos: {len(df)}\n"
        stats_text += (
            f"Avg PCK Range: {df['avg_pck'].min():.3f} - {df['avg_pck'].max():.3f}\n"
        )
        stats_text += f"Avg Brightness Range: {df['avg_brightness'].min():.1f} - {df['avg_brightness'].max():.1f}"

        plt.text(
            0.05,
            0.05,
            stats_text,
            transform=plt.gca().transAxes,
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
            verticalalignment="bottom",
        )

        if self.save_plots and self.output_dir:
            filename = os.path.join(
                self.output_dir, "combined_avg_pck_brightness_per_video.png"
            )
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            print(f"   ✅ Saved: {filename}")
        else:
            print("   ⚠️  Plot saving disabled or no output directory specified")

        try:
            plt.show(block=False)
            print("   ✅ Combined average plot displayed successfully")
        except Exception as e:
            print(f"   ⚠️  Could not display combined average plot: {e}")

        plt.pause(2)
        plt.close()

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

        # Ensure we can display plots
        plt.ioff()  # Turn off interactive mode initially

        # Debug: Print structure of analysis results
        print(f"   Debug: Analysis results contains {len(analysis_results)} videos")
        for vid_name, vid_data in list(analysis_results.items())[
            :1
        ]:  # Show first video structure
            print(f"   Debug: Video '{vid_name}' has keys: {list(vid_data.keys())}")
            for key, value in vid_data.items():
                if key not in [
                    "video_name",
                    "total_frames",
                    "joints_analyzed",
                    "brightness_summary",
                ]:
                    print(
                        f"      PCK column '{key}': {type(value)} with keys: {list(value.keys()) if isinstance(value, dict) else 'Not a dict'}"
                    )
                    if isinstance(value, dict):
                        pck_scores = value.get("pck_scores", [])
                        brightness_vals = value.get("brightness_values", [])
                        print(
                            f"         pck_scores length: {len(pck_scores)}, brightness_values length: {len(brightness_vals)}"
                        )

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
            print(f"   ✅ Saved combined plot: {filename}")
        else:
            print("   ⚠️  Plot saving disabled or no output directory specified")

        # Force display of the plot
        try:
            plt.show(block=False)  # Non-blocking show
            print("   ✅ Plot displayed successfully")
        except Exception as e:
            print(f"   ⚠️  Could not display plot: {e}")

        # Keep the plot open for a bit
        plt.pause(2)
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

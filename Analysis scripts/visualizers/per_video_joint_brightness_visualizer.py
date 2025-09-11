"""
Per-Video Joint Brightness Visualizer

Creates visualizations for per-video joint brightness analysis results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional
import os
from core.base_classes import BaseVisualizer


class PerVideoJointBrightnessVisualizer(BaseVisualizer):
    """Visualizer for per-video joint brightness analysis results."""

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
        """Create all visualization types for per-video analysis results.

        Args:
            analysis_results: Results from PerVideoJointBrightnessAnalyzer
        """
        print("Creating per-video joint brightness visualizations...")

        if not analysis_results:
            print("❌ No analysis results to visualize")
            return

        # Create different visualization types
        self.create_video_comparison_plot(analysis_results)
        self.create_joint_brightness_heatmap(analysis_results)
        self.create_correlation_analysis_plot(analysis_results)
        self.create_per_video_joint_plots(analysis_results)
        self.create_brightness_distribution_plots(analysis_results)
        self.create_video_summary_dashboard(analysis_results)

        print("✅ All per-video visualizations completed")

    def create_video_comparison_plot(self, analysis_results: Dict[str, Any]) -> None:
        """Create comparison plot across videos showing joint brightness patterns."""

        print("Creating video comparison plot...")

        # Prepare data for comparison
        video_data = []

        for video_name, video_results in analysis_results.items():
            brightness_summary = video_results.get("brightness_summary", {})

            for joint_name, joint_stats in brightness_summary.items():
                video_data.append(
                    {
                        "video": video_name,
                        "joint": joint_name,
                        "mean_brightness": joint_stats["mean"],
                        "std_brightness": joint_stats["std"],
                        "valid_frames": joint_stats["valid_frames"],
                    }
                )

        if not video_data:
            print("❌ No brightness data found for comparison")
            return

        df = pd.DataFrame(video_data)

        # Create subplot layout
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(
            "Video Comparison: Joint Brightness Analysis",
            fontsize=16,
            fontweight="bold",
        )

        # 1. Mean brightness comparison across videos
        ax1 = axes[0, 0]
        pivot_mean = df.pivot(index="video", columns="joint", values="mean_brightness")
        sns.heatmap(pivot_mean, annot=True, cmap="viridis", ax=ax1, fmt=".1f")
        ax1.set_title("Mean Brightness by Video and Joint")
        ax1.set_xlabel("Joint")
        ax1.set_ylabel("Video")

        # 2. Brightness variability (std) across videos
        ax2 = axes[0, 1]
        pivot_std = df.pivot(index="video", columns="joint", values="std_brightness")
        sns.heatmap(pivot_std, annot=True, cmap="plasma", ax=ax2, fmt=".1f")
        ax2.set_title("Brightness Variability by Video and Joint")
        ax2.set_xlabel("Joint")
        ax2.set_ylabel("Video")

        # 3. Bar plot: Average brightness per video (across all joints)
        ax3 = axes[1, 0]
        video_avg = (
            df.groupby("video")["mean_brightness"].mean().sort_values(ascending=False)
        )
        video_avg.plot(kind="bar", ax=ax3, color="skyblue")
        ax3.set_title("Average Brightness per Video (All Joints)")
        ax3.set_xlabel("Video")
        ax3.set_ylabel("Mean Brightness")
        ax3.tick_params(axis="x", rotation=45)

        # 4. Bar plot: Average brightness per joint (across all videos)
        ax4 = axes[1, 1]
        joint_avg = (
            df.groupby("joint")["mean_brightness"].mean().sort_values(ascending=False)
        )
        joint_avg.plot(kind="bar", ax=ax4, color="lightcoral")
        ax4.set_title("Average Brightness per Joint (All Videos)")
        ax4.set_xlabel("Joint")
        ax4.set_ylabel("Mean Brightness")
        ax4.tick_params(axis="x", rotation=45)

        plt.tight_layout()

        if self.save_plots and self.output_dir:
            filename = os.path.join(self.output_dir, "video_comparison_plot.png")
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            print(f"   Saved: {filename}")

        plt.show()

    def create_joint_brightness_heatmap(self, analysis_results: Dict[str, Any]) -> None:
        """Create heatmap showing brightness patterns across videos and joints."""

        print("Creating joint brightness heatmap...")

        # Prepare data
        videos = list(analysis_results.keys())

        # Get all joint names
        all_joints = set()
        for video_results in analysis_results.values():
            brightness_summary = video_results.get("brightness_summary", {})
            all_joints.update(brightness_summary.keys())

        all_joints = sorted(list(all_joints))

        # Create brightness matrix
        brightness_matrix = np.zeros((len(videos), len(all_joints)))

        for i, video_name in enumerate(videos):
            video_results = analysis_results[video_name]
            brightness_summary = video_results.get("brightness_summary", {})

            for j, joint_name in enumerate(all_joints):
                if joint_name in brightness_summary:
                    brightness_matrix[i, j] = brightness_summary[joint_name]["mean"]
                else:
                    brightness_matrix[i, j] = np.nan

        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            brightness_matrix,
            xticklabels=all_joints,
            yticklabels=videos,
            annot=True,
            cmap="viridis",
            fmt=".1f",
            cbar_kws={"label": "Mean Brightness"},
        )
        plt.title(
            "Joint Brightness Heatmap Across Videos", fontsize=14, fontweight="bold"
        )
        plt.xlabel("Joints")
        plt.ylabel("Videos")
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()

        if self.save_plots and self.output_dir:
            filename = os.path.join(self.output_dir, "joint_brightness_heatmap.png")
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            print(f"   Saved: {filename}")

        plt.show()

    def create_correlation_analysis_plot(
        self, analysis_results: Dict[str, Any]
    ) -> None:
        """Create correlation analysis plots showing PCK-brightness relationships."""

        print("Creating correlation analysis plot...")

        # Collect correlation data
        correlation_data = []

        for video_name, video_results in analysis_results.items():
            for pck_column, pck_results in video_results.items():
                if pck_column in [
                    "video_name",
                    "total_frames",
                    "joints_analyzed",
                    "brightness_summary",
                ]:
                    continue

                if "correlation" in pck_results:
                    correlation_data.append(
                        {
                            "video": video_name,
                            "joint": pck_results.get("joint_name", "unknown"),
                            "threshold": pck_results.get("threshold", "unknown"),
                            "pearson_correlation": pck_results["correlation"][
                                "pearson"
                            ],
                            "valid_frames": pck_results.get("valid_frames", 0),
                        }
                    )

        if not correlation_data:
            print("❌ No correlation data found")
            return

        df = pd.DataFrame(correlation_data)

        # Create subplot layout
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(
            "PCK-Brightness Correlation Analysis Across Videos",
            fontsize=16,
            fontweight="bold",
        )

        # 1. Correlation heatmap by video and joint
        ax1 = axes[0, 0]
        if len(df) > 0:
            pivot_corr = df.pivot_table(
                index="video",
                columns="joint",
                values="pearson_correlation",
                aggfunc="mean",
            )
            sns.heatmap(
                pivot_corr,
                annot=True,
                cmap="RdBu_r",
                center=0,
                ax=ax1,
                fmt=".3f",
                vmin=-1,
                vmax=1,
            )
            ax1.set_title("Mean Correlation by Video and Joint")

        # 2. Distribution of correlations
        ax2 = axes[0, 1]
        ax2.hist(
            df["pearson_correlation"],
            bins=20,
            alpha=0.7,
            color="skyblue",
            edgecolor="black",
        )
        ax2.axvline(x=0, color="red", linestyle="--", alpha=0.7, label="No Correlation")
        ax2.set_xlabel("Pearson Correlation")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Distribution of PCK-Brightness Correlations")
        ax2.legend()

        # 3. Correlation by joint (across all videos)
        ax3 = axes[1, 0]
        joint_corr = df.groupby("joint")["pearson_correlation"].mean().sort_values()
        joint_corr.plot(kind="barh", ax=ax3, color="lightgreen")
        ax3.set_xlabel("Mean Pearson Correlation")
        ax3.set_title("Average Correlation by Joint")
        ax3.axvline(x=0, color="red", linestyle="--", alpha=0.7)

        # 4. Correlation by video (across all joints)
        ax4 = axes[1, 1]
        video_corr = df.groupby("video")["pearson_correlation"].mean().sort_values()
        video_corr.plot(kind="barh", ax=ax4, color="lightcoral")
        ax4.set_xlabel("Mean Pearson Correlation")
        ax4.set_title("Average Correlation by Video")
        ax4.axvline(x=0, color="red", linestyle="--", alpha=0.7)

        plt.tight_layout()

        if self.save_plots and self.output_dir:
            filename = os.path.join(self.output_dir, "correlation_analysis_plot.png")
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            print(f"   Saved: {filename}")

        plt.show()

    def create_per_video_joint_plots(self, analysis_results: Dict[str, Any]) -> None:
        """Create individual plots for each video showing all joints."""

        print("Creating per-video joint plots...")

        for video_name, video_results in analysis_results.items():
            print(f"   Creating plot for video: {video_name}")

            # Get joints with PCK data
            joint_data = {}
            for pck_column, pck_results in video_results.items():
                if pck_column in [
                    "video_name",
                    "total_frames",
                    "joints_analyzed",
                    "brightness_summary",
                ]:
                    continue

                joint_name = pck_results.get("joint_name", "unknown")
                if joint_name not in joint_data:
                    joint_data[joint_name] = {}

                threshold = pck_results.get("threshold", "unknown")
                joint_data[joint_name][threshold] = {
                    "pck_scores": pck_results.get("pck_scores", []),
                    "brightness_values": pck_results.get("brightness_values", []),
                    "correlation": pck_results.get("correlation", {}).get(
                        "pearson", 0.0
                    ),
                }

            if not joint_data:
                continue

            # Create subplots for this video
            n_joints = len(joint_data)
            cols = min(3, n_joints)
            rows = (n_joints + cols - 1) // cols

            fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
            if rows == 1 and cols == 1:
                axes = [axes]
            elif rows == 1:
                axes = axes
            else:
                axes = axes.flatten()

            fig.suptitle(
                f"Joint Analysis for Video: {video_name}",
                fontsize=14,
                fontweight="bold",
            )

            for idx, (joint_name, thresholds) in enumerate(joint_data.items()):
                if idx >= len(axes):
                    break

                ax = axes[idx]

                # Plot scatter for first threshold (or all if multiple)
                colors = plt.cm.tab10(np.linspace(0, 1, len(thresholds)))

                for i, (threshold, data) in enumerate(thresholds.items()):
                    pck_scores = data["pck_scores"]
                    brightness_values = data["brightness_values"]
                    correlation = data["correlation"]

                    if len(pck_scores) > 0 and len(brightness_values) > 0:
                        ax.scatter(
                            brightness_values,
                            pck_scores,
                            alpha=0.6,
                            s=20,
                            color=colors[i],
                            label=f"Thr {threshold} (r={correlation:.3f})",
                        )

                ax.set_xlabel("Brightness")
                ax.set_ylabel("PCK Score")
                ax.set_title(f"{joint_name}")
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)

            # Hide unused subplots
            for idx in range(len(joint_data), len(axes)):
                axes[idx].set_visible(False)

            plt.tight_layout()

            if self.save_plots and self.output_dir:
                video_safe_name = str(video_name).replace("/", "_").replace("\\", "_")
                filename = os.path.join(
                    self.output_dir, f"video_{video_safe_name}_joints.png"
                )
                plt.savefig(filename, dpi=300, bbox_inches="tight")
                print(f"   Saved: {filename}")
            else:
                plt.show()

            plt.close()

    def create_brightness_distribution_plots(
        self, analysis_results: Dict[str, Any]
    ) -> None:
        """Create brightness distribution plots across videos and joints."""

        print("Creating brightness distribution plots...")

        # Collect brightness data
        brightness_data = []

        for video_name, video_results in analysis_results.items():
            brightness_summary = video_results.get("brightness_summary", {})

            for joint_name, joint_stats in brightness_summary.items():
                brightness_data.append(
                    {
                        "video": video_name,
                        "joint": joint_name,
                        "mean_brightness": joint_stats["mean"],
                        "std_brightness": joint_stats["std"],
                        "min_brightness": joint_stats["min"],
                        "max_brightness": joint_stats["max"],
                    }
                )

        if not brightness_data:
            print("❌ No brightness distribution data found")
            return

        df = pd.DataFrame(brightness_data)

        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("Brightness Distribution Analysis", fontsize=16, fontweight="bold")

        # 1. Box plot by joint
        ax1 = axes[0, 0]
        sns.boxplot(data=df, x="joint", y="mean_brightness", ax=ax1)
        ax1.set_title("Brightness Distribution by Joint")
        ax1.set_xlabel("Joint")
        ax1.set_ylabel("Mean Brightness")
        ax1.tick_params(axis="x", rotation=45)

        # 2. Box plot by video
        ax2 = axes[0, 1]
        sns.boxplot(data=df, x="video", y="mean_brightness", ax=ax2)
        ax2.set_title("Brightness Distribution by Video")
        ax2.set_xlabel("Video")
        ax2.set_ylabel("Mean Brightness")
        ax2.tick_params(axis="x", rotation=45)

        # 3. Scatter plot: mean vs std brightness
        ax3 = axes[1, 0]
        colors = plt.cm.tab10(np.linspace(0, 1, len(df["joint"].unique())))
        joint_colors = {
            joint: color for joint, color in zip(df["joint"].unique(), colors)
        }

        for joint in df["joint"].unique():
            joint_data = df[df["joint"] == joint]
            ax3.scatter(
                joint_data["mean_brightness"],
                joint_data["std_brightness"],
                label=joint,
                color=joint_colors[joint],
                alpha=0.7,
                s=50,
            )

        ax3.set_xlabel("Mean Brightness")
        ax3.set_ylabel("Brightness Standard Deviation")
        ax3.set_title("Brightness Mean vs Variability")
        ax3.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax3.grid(True, alpha=0.3)

        # 4. Range plot (min to max brightness)
        ax4 = axes[1, 1]
        for i, joint in enumerate(df["joint"].unique()):
            joint_data = df[df["joint"] == joint]
            for _, row in joint_data.iterrows():
                ax4.plot(
                    [row["min_brightness"], row["max_brightness"]],
                    [i, i],
                    marker="o",
                    linewidth=2,
                    alpha=0.7,
                    color=joint_colors[joint],
                )

        ax4.set_yticks(range(len(df["joint"].unique())))
        ax4.set_yticklabels(df["joint"].unique())
        ax4.set_xlabel("Brightness Range")
        ax4.set_title("Brightness Range by Joint")
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if self.save_plots and self.output_dir:
            filename = os.path.join(
                self.output_dir, "brightness_distribution_plots.png"
            )
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            print(f"   Saved: {filename}")

        plt.show()

    def create_video_summary_dashboard(self, analysis_results: Dict[str, Any]) -> None:
        """Create a comprehensive dashboard summarizing all videos."""

        print("Creating video summary dashboard...")

        # Prepare summary data
        video_summaries = []

        for video_name, video_results in analysis_results.items():
            total_frames = video_results.get("total_frames", 0)
            joints_analyzed = video_results.get("joints_analyzed", [])
            brightness_summary = video_results.get("brightness_summary", {})

            # Calculate overall video metrics
            all_brightness_means = [
                stats["mean"]
                for stats in brightness_summary.values()
                if stats["mean"] > 0
            ]
            avg_brightness = (
                np.mean(all_brightness_means) if all_brightness_means else 0
            )

            # Count valid correlations
            valid_correlations = []
            for pck_column, pck_results in video_results.items():
                if pck_column in [
                    "video_name",
                    "total_frames",
                    "joints_analyzed",
                    "brightness_summary",
                ]:
                    continue
                if "correlation" in pck_results:
                    corr = pck_results["correlation"]["pearson"]
                    if not np.isnan(corr):
                        valid_correlations.append(corr)

            avg_correlation = np.mean(valid_correlations) if valid_correlations else 0

            video_summaries.append(
                {
                    "video": video_name,
                    "total_frames": total_frames,
                    "joints_count": len(joints_analyzed),
                    "avg_brightness": avg_brightness,
                    "avg_correlation": avg_correlation,
                    "strong_correlations": sum(
                        1 for c in valid_correlations if abs(c) > 0.5
                    ),
                }
            )

        df = pd.DataFrame(video_summaries)

        # Create dashboard
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle("Video Analysis Summary Dashboard", fontsize=16, fontweight="bold")

        # 1. Video metrics bar chart
        ax1 = axes[0, 0]
        x_pos = np.arange(len(df))
        ax1.bar(x_pos, df["total_frames"], alpha=0.7, color="skyblue")
        ax1.set_xlabel("Video")
        ax1.set_ylabel("Total Frames")
        ax1.set_title("Video Frame Counts")
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(df["video"], rotation=45, ha="right")

        # 2. Average brightness per video
        ax2 = axes[0, 1]
        bars = ax2.bar(x_pos, df["avg_brightness"], alpha=0.7, color="lightgreen")
        ax2.set_xlabel("Video")
        ax2.set_ylabel("Average Brightness")
        ax2.set_title("Average Brightness per Video")
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(df["video"], rotation=45, ha="right")

        # Color code bars by brightness level
        brightness_norm = plt.Normalize(
            vmin=df["avg_brightness"].min(), vmax=df["avg_brightness"].max()
        )
        for bar, brightness in zip(bars, df["avg_brightness"]):
            bar.set_color(plt.cm.viridis(brightness_norm(brightness)))

        # 3. Average correlation per video
        ax3 = axes[0, 2]
        colors = [
            "red" if c < -0.3 else "orange" if c < 0.3 else "green"
            for c in df["avg_correlation"]
        ]
        ax3.bar(x_pos, df["avg_correlation"], alpha=0.7, color=colors)
        ax3.axhline(y=0, color="black", linestyle="--", alpha=0.5)
        ax3.set_xlabel("Video")
        ax3.set_ylabel("Average Correlation")
        ax3.set_title("Average PCK-Brightness Correlation")
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(df["video"], rotation=45, ha="right")

        # 4. Strong correlations count
        ax4 = axes[1, 0]
        ax4.bar(x_pos, df["strong_correlations"], alpha=0.7, color="purple")
        ax4.set_xlabel("Video")
        ax4.set_ylabel("Strong Correlations Count")
        ax4.set_title("Strong Correlations (|r| > 0.5) per Video")
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(df["video"], rotation=45, ha="right")

        # 5. Scatter: brightness vs correlation
        ax5 = axes[1, 1]
        scatter = ax5.scatter(
            df["avg_brightness"],
            df["avg_correlation"],
            s=df["total_frames"] / 10,  # Size by frame count
            alpha=0.7,
            c=df["joints_count"],
            cmap="plasma",
        )
        ax5.set_xlabel("Average Brightness")
        ax5.set_ylabel("Average Correlation")
        ax5.set_title("Brightness vs Correlation (Size = Frames)")
        plt.colorbar(scatter, ax=ax5, label="Joints Count")
        ax5.grid(True, alpha=0.3)

        # Add video labels
        for _, row in df.iterrows():
            ax5.annotate(
                row["video"],
                (row["avg_brightness"], row["avg_correlation"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                alpha=0.8,
            )

        # 6. Summary statistics table
        ax6 = axes[1, 2]
        ax6.axis("off")

        # Create summary statistics
        summary_stats = [
            ["Total Videos", len(df)],
            ["Avg Frames/Video", f"{df['total_frames'].mean():.0f}"],
            ["Avg Brightness", f"{df['avg_brightness'].mean():.2f}"],
            ["Avg Correlation", f"{df['avg_correlation'].mean():.3f}"],
            ["Videos with Strong Corr", sum(df["strong_correlations"] > 0)],
            ["Best Brightness Video", df.loc[df["avg_brightness"].idxmax(), "video"]],
            ["Best Correlation Video", df.loc[df["avg_correlation"].idxmax(), "video"]],
        ]

        table = ax6.table(
            cellText=summary_stats,
            colLabels=["Metric", "Value"],
            cellLoc="left",
            loc="center",
            bbox=[0, 0, 1, 1],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        ax6.set_title("Summary Statistics")

        plt.tight_layout()

        if self.save_plots and self.output_dir:
            filename = os.path.join(self.output_dir, "video_summary_dashboard.png")
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

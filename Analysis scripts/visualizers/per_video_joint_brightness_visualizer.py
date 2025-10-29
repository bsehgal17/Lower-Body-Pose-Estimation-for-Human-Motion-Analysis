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

        # Create individual scatter plots for each video (this is what the user requested)
        self.create_per_video_scatter_plots(analysis_results)

        # Create combined scatter plot showing all videos and joints together (detailed plot)
        if self.create_individual_plots:
            self.create_combined_scatter_plot(analysis_results)

        print("✅ Per-video visualization completed")

    def create_pck_brightness_plot(self, analysis_results: dict) -> None:
        print("Creating combined average PCK vs brightness plot per video...")

        plot_data = []
        for video_name, video_results in analysis_results.items():
            # Extract PCK scores and brightness values
            pck_dict = video_results.get("pck_scores", {})
            brightness_dict = video_results.get("avg_brightness", {})

            # Convert dicts to lists
            pck_values = list(pck_dict.values())
            brightness_values = list(brightness_dict.values())

            # Compute averages
            if pck_values and brightness_values:
                plot_data.append(
                    {
                        "video": video_name,
                        "avg_pck": np.mean(pck_values),
                        "avg_brightness": np.mean(brightness_values),
                    }
                )

        if not plot_data:
            print("❌ No data to plot.")
            return

        df = pd.DataFrame(plot_data)
        fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)

        # Unique videos
        videos = df["video"].unique()
        import seaborn as sns

        colors = sns.color_palette("husl", n_colors=len(videos))

        # Scatter plot per video
        for i, video in enumerate(videos):
            row = df[df["video"] == video].iloc[0]
            ax.scatter(
                row["avg_brightness"],
                row["avg_pck"],
                label=video,
                color=colors[i],
                s=120,
                edgecolors="black",
                linewidth=0.7,
                alpha=0.9,
            )

        # Trend line
        if len(df) > 2:
            z = np.polyfit(df["avg_brightness"], df["avg_pck"], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(
                df["avg_brightness"].min(), df["avg_brightness"].max(), 100
            )
            ax.plot(
                x_trend, p(x_trend), "r--", alpha=0.7, linewidth=2, label="Trend line"
            )

        ax.set_xlabel("Average Brightness", fontsize=12)
        ax.set_ylabel("Average PCK Score", fontsize=12)
        ax.set_title(
            "Average PCK vs Brightness Per Video", fontsize=14, fontweight="bold"
        )
        ax.grid(True, alpha=0.3)

        # Legend outside, smaller font, multiple columns
        num_cols = min(len(videos), 2)  # max 4 columns
        plt.subplots_adjust(right=0.75)  # leave room for legend
        ax.legend(
            bbox_to_anchor=(1.01, 1),
            loc="upper left",
            fontsize=7,
            ncol=num_cols,
            frameon=True,
        )

        if (
            hasattr(self, "save_plots")
            and self.save_plots
            and hasattr(self, "output_dir")
            and self.output_dir
        ):
            filename = os.path.join(
                self.output_dir, "combined_avg_pck_brightness_per_video.png"
            )
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            print(f"✅ Saved: {filename}")

        plt.show()
        plt.close()

    def create_per_video_scatter_plots(self, analysis_results: Dict[str, Any]) -> None:
        """Create individual scatter plots for each video showing PCK vs brightness for all joints.

        This creates separate scatter plots for each video with video information prominently displayed.
        Each plot shows PCK vs brightness where brightness is taken around joint from ground truth coordinates.
        """
        print(
            "Creating per-video scatter plots (PCK vs brightness from GT joint locations)..."
        )

        for video_name, video_results in analysis_results.items():
            print(f"   Creating scatter plot for video: {video_name}")

            # Extract video metadata for display
            video_metadata = self._extract_video_metadata(video_name, video_results)

            # Get PCK scores and brightness values for this video
            pck_scores = video_results.get("pck_scores", {})
            avg_brightness = video_results.get("avg_brightness", {})

            if not pck_scores or not avg_brightness:
                print(f"   ❌ No PCK scores or brightness data found for {video_name}")
                continue

            # Prepare data for plotting - using average values per joint
            plot_data = []
            for pck_column, pck_value in pck_scores.items():
                # Parse joint name and threshold from column name
                joint_name, threshold = self._parse_pck_column_name(pck_column)

                # Get corresponding brightness value
                if joint_name in avg_brightness:
                    brightness_value = avg_brightness[joint_name]
                    plot_data.append(
                        {
                            "joint": joint_name,
                            "threshold": threshold,
                            "pck": pck_value,
                            "brightness": brightness_value,
                            "pck_column": pck_column,
                        }
                    )

            if not plot_data:
                print(f"   ❌ No matching PCK/brightness data found for {video_name}")
                continue

            # Create the scatter plot for this video
            df = pd.DataFrame(plot_data)

            # Group by threshold for subplots
            unique_thresholds = sorted(df["threshold"].unique())
            n_thresholds = len(unique_thresholds)

            if n_thresholds == 1:
                fig, ax = plt.subplots(figsize=(14, 10))
                axes = [ax]
            else:
                fig, axes = plt.subplots(
                    1, n_thresholds, figsize=(7 * n_thresholds, 10)
                )
                if n_thresholds == 1:
                    axes = [axes]

            # Color palette for joints
            unique_joints = df["joint"].unique()
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
                            alpha=0.7,
                            s=80,
                            edgecolors="black",
                            linewidth=1,
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
                            f"Correlation: r = {correlation:.3f}",
                            transform=ax.transAxes,
                            fontsize=12,
                            fontweight="bold",
                            bbox=dict(
                                boxstyle="round,pad=0.4",
                                facecolor="lightblue",
                                alpha=0.8,
                            ),
                        )

                # Add trend line if we have enough data points
                if len(threshold_data) > 2:
                    z = np.polyfit(
                        threshold_data["brightness"], threshold_data["pck"], 1
                    )
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

                ax.set_xlabel(
                    "Average Brightness (from GT Joint Location)", fontsize=12
                )
                ax.set_ylabel("PCK Score", fontsize=12)
                ax.set_title(
                    f"PCK vs Brightness - Threshold {threshold}",
                    fontsize=14,
                    fontweight="bold",
                )
                ax.grid(True, alpha=0.3)
                ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)

                # Set reasonable axis limits
                ax.set_xlim(
                    threshold_data["brightness"].min() * 0.95,
                    threshold_data["brightness"].max() * 1.05,
                )
                ax.set_ylim(-0.05, 1.05)

            # Create comprehensive title with video information
            title_text = (
                f"PCK vs Joint Brightness Analysis\n{video_metadata['display_title']}"
            )
            plt.suptitle(title_text, fontsize=16, fontweight="bold", y=0.98)

            # Add video metadata as text annotation
            metadata_text = self._format_video_metadata(video_metadata)

            # Place metadata on the figure (adjust position based on subplot layout)
            if n_thresholds == 1:
                fig.text(
                    0.02,
                    0.02,
                    metadata_text,
                    fontsize=10,
                    bbox=dict(
                        boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8
                    ),
                    verticalalignment="bottom",
                )
            else:
                fig.text(
                    0.02,
                    0.02,
                    metadata_text,
                    fontsize=9,
                    bbox=dict(
                        boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8
                    ),
                    verticalalignment="bottom",
                )

            plt.tight_layout()
            plt.subplots_adjust(bottom=0.15)  # Make room for metadata

            # Save the plot
            if self.save_plots and self.output_dir:
                clean_video_name = self._clean_video_name_for_filename(video_name)
                filename = os.path.join(
                    self.output_dir, f"individual_pck_brightness_{clean_video_name}.png"
                )
                plt.savefig(filename, dpi=300, bbox_inches="tight")
                print(f"   ✅ Saved: {filename}")

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
                video_summary[f"{joint_name}_mean_brightness"] = stats.get("mean", 0.0)
                video_summary[f"{joint_name}_std_brightness"] = stats.get("std", 0.0)
                video_summary[f"{joint_name}_valid_frames"] = stats.get(
                    "valid_frames", 0
                )

            video_summaries.append(video_summary)

        video_df = pd.DataFrame(video_summaries)
        video_csv_path = os.path.join(self.output_dir, "video_brightness_summary.csv")
        video_df.to_csv(video_csv_path, index=False)
        print(f"   Saved video summary: {video_csv_path}")

        # 2. Detailed results CSV (joint-wise)
        detailed_results = []
        for video_name, video_results in analysis_results.items():
            # Only consider pck_scores if they exist
            pck_scores = video_results.get("pck_scores", {})

            if isinstance(pck_scores, dict):
                for joint_key, pck_value in pck_scores.items():
                    # Handle numbers directly
                    if isinstance(pck_value, (int, float)):
                        result_row = {
                            "video_name": video_name,
                            "joint_metric": joint_key,
                            "pck_value": pck_value,
                        }
                        detailed_results.append(result_row)
                    # If future format uses dict, handle safely
                    elif isinstance(pck_value, dict):
                        result_row = {
                            "video_name": video_name,
                            "joint_metric": joint_key,
                            "pck_value": pck_value.get("value", None),
                            "valid_frames": pck_value.get("valid_frames", None),
                            "threshold": pck_value.get("threshold", None),
                        }
                        detailed_results.append(result_row)

        if detailed_results:
            detailed_df = pd.DataFrame(detailed_results)
            detailed_csv_path = os.path.join(
                self.output_dir, "detailed_pck_brightness_results.csv"
            )
            detailed_df.to_csv(detailed_csv_path, index=False)
            print(f"   Saved detailed results: {detailed_csv_path}")
        else:
            print("❌ No detailed PCK results found.")

        print("✅ CSV export completed")

    def _extract_video_metadata(
        self, video_name: str, video_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract and organize video metadata for display."""
        metadata = {
            "video_name": video_name,
            "display_title": f"Video: {video_name}",
            "total_frames": video_results.get("total_frames", "N/A"),
            "synced_frames": video_results.get("synced_frames", "N/A"),
            "sync_offset": video_results.get("sync_offset", 0),
            "joints_analyzed": len(video_results.get("joints_analyzed", [])),
            "joint_names": video_results.get("joints_analyzed", []),
            "num_pck_metrics": len(
                [
                    k
                    for k in video_results.keys()
                    if k
                    not in [
                        "video_name",
                        "total_frames",
                        "synced_frames",
                        "sync_offset",
                        "joints_analyzed",
                        "avg_brightness",
                        "pck_scores",
                        "brightness_summary",
                    ]
                ]
            ),
        }

        # Calculate average brightness across all joints
        avg_brightness_values = list(video_results.get("avg_brightness", {}).values())
        if avg_brightness_values:
            metadata["avg_brightness_all_joints"] = np.mean(avg_brightness_values)
            metadata["brightness_range"] = (
                min(avg_brightness_values),
                max(avg_brightness_values),
            )
        else:
            metadata["avg_brightness_all_joints"] = "N/A"
            metadata["brightness_range"] = ("N/A", "N/A")

        # Calculate average PCK across all metrics
        pck_scores = list(video_results.get("pck_scores", {}).values())
        if pck_scores:
            metadata["avg_pck_all_metrics"] = np.mean(pck_scores)
            metadata["pck_range"] = (min(pck_scores), max(pck_scores))
        else:
            metadata["avg_pck_all_metrics"] = "N/A"
            metadata["pck_range"] = ("N/A", "N/A")

        return metadata

    def _format_video_metadata(self, metadata: Dict[str, Any]) -> str:
        """Format video metadata for display as text annotation."""
        lines = [
            f"📹 Video: {metadata['video_name']}",
            f"🎞️ Total Frames: {metadata['total_frames']} | Synced: {metadata['synced_frames']} | Offset: {metadata['sync_offset']}",
            f"🎯 Joints Analyzed: {metadata['joints_analyzed']} ({', '.join(metadata['joint_names'])})",
        ]

        # Add brightness info
        if metadata["avg_brightness_all_joints"] != "N/A":
            lines.append(
                f"💡 Avg Brightness: {metadata['avg_brightness_all_joints']:.1f} "
                f"(Range: {metadata['brightness_range'][0]:.1f}-{metadata['brightness_range'][1]:.1f})"
            )

        # Add PCK info
        if metadata["avg_pck_all_metrics"] != "N/A":
            lines.append(
                f"📊 Avg PCK: {metadata['avg_pck_all_metrics']:.3f} "
                f"(Range: {metadata['pck_range'][0]:.3f}-{metadata['pck_range'][1]:.3f})"
            )

        return "\n".join(lines)

    def _parse_pck_column_name(self, column_name: str) -> tuple:
        """Parse joint name and threshold from PCK column name."""
        # Format: LEFT_HIP_jointwise_pck_0.01
        parts = column_name.split("_")

        # Find the threshold (last part after splitting by _)
        threshold = parts[-1] if parts else "0.05"

        # Joint name is everything before 'jointwise'
        joint_parts = []
        for part in parts:
            if part.lower() == "jointwise":
                break
            joint_parts.append(part)

        joint_name = "_".join(joint_parts) if joint_parts else "UNKNOWN"

        return joint_name, threshold

    def _clean_video_name_for_filename(self, video_name: str) -> str:
        """Clean video name to make it suitable for filenames.

        Handles special cases like HumanEva format S1_Walking_(C1).
        """
        clean_name = (
            str(video_name)
            .replace("/", "_")
            .replace("\\", "_")
            .replace(":", "_")
            .replace("(", "")
            .replace(")", "")
            .replace(" ", "_")
        )
        return clean_name

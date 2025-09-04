"""
Brightness Extractor Script

Extracts brightness values from video frames for specific PCK scores.
Focus: Video brightness extraction only.
"""

# from pck_analysis.pck_score_filter import PCKScoreFilter
from processors import VideoPathResolver, FrameSynchronizer
from extractors import MetricExtractorFactory
from config import ConfigManager
import sys
import os
import pandas as pd
import numpy as np
from typing import List, Dict

# Add the Analysis scripts directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class BrightnessExtractor:
    """Extract brightness values for specific PCK scores."""

    def __init__(self, dataset_name: str):
        """Initialize with dataset name."""
        self.dataset_name = dataset_name
        self.config = ConfigManager.load_config(dataset_name)
        self.path_resolver = VideoPathResolver(self.config)
        self.frame_sync = FrameSynchronizer(self.config)
        from pck_analysis.pck_score_filter import PCKScoreFilter

        self.score_filter = PCKScoreFilter(dataset_name)

    def extract_brightness_for_scores(
        self, target_scores: List[int], pck_threshold: str = None
    ) -> Dict[int, List[float]]:
        """Extract brightness values for specific PCK scores."""
        print(f"Extracting brightness for PCK scores: {target_scores}")
        print("=" * 60)

        # Load and filter data
        filtered_data = self.score_filter.filter_by_specific_scores(
            target_scores, pck_threshold
        )
        if filtered_data is None or filtered_data.empty:
            print("[ERROR] No data found for specified scores")
            return {}

        # Group by video/camera combination
        grouping_cols = self.config.get_grouping_columns()
        if not grouping_cols:
            print("[ERROR] No grouping columns available")
            return {}

        brightness_by_score = {score: [] for score in target_scores}
        grouped_data = filtered_data.groupby(grouping_cols)

        print(f"Processing {len(grouped_data)} video groups...")

        for group_name, group_data in grouped_data:
            # Create video row data for path resolution
            video_row_data = {
                col: group_name[grouping_cols.index(col)] for col in grouping_cols
            }
            video_row = pd.Series(video_row_data)

            # Find video path
            video_path = self.path_resolver.find_video_for_row(video_row)

            if not video_path or not os.path.exists(video_path):
                video_info = ", ".join([f"{k}: {v}" for k, v in video_row_data.items()])
                print(f"⚠️  Video not found: {video_info}")
                continue

            print(f"📹 Processing: {os.path.basename(video_path)}")

            # Extract brightness for all frames
            brightness_extractor = MetricExtractorFactory.create_extractor(
                "brightness", video_path
            )
            brightness_data = brightness_extractor.extract()

            if not brightness_data:
                print("   ❌ Failed to extract brightness")
                continue

            # Get synchronization offset
            synced_start_frame = self.frame_sync.get_synced_start_frame(video_row_data)
            brightness_data_sliced = brightness_data[synced_start_frame:]

            # Process each frame in this group
            for _, frame_row in group_data.iterrows():
                frame_idx = int(frame_row["frame_idx"])

                # Check each PCK threshold column
                for pck_col in self.config.pck_per_frame_score_columns:
                    if pck_threshold and pck_col != pck_threshold:
                        continue

                    if pck_col in frame_row:
                        pck_score = int(round(frame_row[pck_col]))

                        if pck_score in target_scores and frame_idx < len(
                            brightness_data_sliced
                        ):
                            frame_brightness = brightness_data_sliced[frame_idx]
                            brightness_by_score[pck_score].append(frame_brightness)

        # Print summary
        print("\n📊 Extraction Summary:")
        for score in target_scores:
            count = len(brightness_by_score[score])
            if count > 0:
                mean_brightness = np.mean(brightness_by_score[score])
                print(
                    f"  PCK {score}: {count} frames, avg brightness: {mean_brightness:.1f}"
                )
            else:
                print(f"  PCK {score}: No frames found")

        return brightness_by_score

    def extract_brightness_for_score_range(
        self, min_score: int, max_score: int, pck_threshold: str = None
    ) -> Dict[int, List[float]]:
        """Extract brightness for a range of PCK scores."""
        target_scores = list(range(min_score, max_score + 1))
        return self.extract_brightness_for_scores(target_scores, pck_threshold)

    def extract_brightness_statistics(
        self, target_scores: List[int], pck_threshold: str = None
    ) -> Dict[int, Dict[str, float]]:
        """Extract brightness and calculate statistics for each score."""
        brightness_data = self.extract_brightness_for_scores(
            target_scores, pck_threshold
        )

        statistics = {}
        for score, brightness_values in brightness_data.items():
            if brightness_values:
                stats = {
                    "count": len(brightness_values),
                    "mean": np.mean(brightness_values),
                    "std": np.std(brightness_values),
                    "min": np.min(brightness_values),
                    "max": np.max(brightness_values),
                    "median": np.median(brightness_values),
                    "q25": np.percentile(brightness_values, 25),
                    "q75": np.percentile(brightness_values, 75),
                }
                statistics[score] = stats

        return statistics

    def compare_brightness_across_scores(
        self, scores: List[int], pck_threshold: str = None
    ) -> Dict[str, any]:
        """Compare brightness patterns across different PCK scores."""
        print(f"Comparing brightness across PCK scores: {scores}")

        brightness_stats = self.extract_brightness_statistics(scores, pck_threshold)

        if not brightness_stats:
            print("❌ No brightness data available for comparison")
            return {}

        comparison = {
            "scores": scores,
            "statistics": brightness_stats,
            "trends": {},
            "differences": {},
        }

        # Calculate trends
        means = [
            brightness_stats[score]["mean"]
            for score in scores
            if score in brightness_stats
        ]
        if len(means) > 1:
            # Linear correlation between score and brightness
            correlation = np.corrcoef(scores[: len(means)], means)[0, 1]
            comparison["trends"]["score_brightness_correlation"] = correlation

            # Brightness increase per score point
            if len(means) > 1:
                brightness_slope = (means[-1] - means[0]) / (scores[-1] - scores[0])
                comparison["trends"]["brightness_per_score_point"] = brightness_slope

        # Calculate pairwise differences
        for i, score1 in enumerate(scores):
            if score1 not in brightness_stats:
                continue
            for score2 in scores[i + 1 :]:
                if score2 not in brightness_stats:
                    continue

                diff_key = f"{score1}_vs_{score2}"
                mean_diff = (
                    brightness_stats[score2]["mean"] - brightness_stats[score1]["mean"]
                )
                comparison["differences"][diff_key] = mean_diff

        # Print comparison summary
        print("\n🔍 Brightness Comparison Results:")
        if "score_brightness_correlation" in comparison["trends"]:
            corr = comparison["trends"]["score_brightness_correlation"]
            print(f"  Score-Brightness Correlation: {corr:.3f}")

        if "brightness_per_score_point" in comparison["trends"]:
            slope = comparison["trends"]["brightness_per_score_point"]
            print(f"  Brightness change per score point: {slope:.2f}")

        print("\n  Mean brightness by score:")
        for score in scores:
            if score in brightness_stats:
                mean_bright = brightness_stats[score]["mean"]
                frame_count = brightness_stats[score]["count"]
                print(f"    PCK {score}: {mean_bright:.1f} ({frame_count} frames)")

        return comparison

    def export_brightness_data(
        self, brightness_data: Dict[int, List[float]], filename: str = None
    ) -> str:
        """Export brightness data to CSV."""
        if not brightness_data:
            print("❌ No brightness data to export")
            return ""

        if filename is None:
            filename = f"brightness_data_{self.dataset_name}.csv"

        # Create DataFrame from brightness data
        export_rows = []
        for score, brightness_values in brightness_data.items():
            for i, brightness in enumerate(brightness_values):
                export_rows.append(
                    {"pck_score": score, "frame_number": i, "brightness": brightness}
                )

        df = pd.DataFrame(export_rows)
        output_path = os.path.join(self.config.save_folder, filename)
        os.makedirs(self.config.save_folder, exist_ok=True)
        df.to_csv(output_path, index=False)

        print(f"✅ Brightness data exported to: {output_path}")
        return output_path

    def create_brightness_histogram(
        self,
        target_scores: List[int],
        pck_threshold: str = None,
        save_plot: bool = True,
    ):
        """Create histogram of brightness values for each score."""
        brightness_data = self.extract_brightness_for_scores(
            target_scores, pck_threshold
        )

        if not brightness_data:
            print("❌ No brightness data for histogram")
            return

        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 8))

        for score, brightness_values in brightness_data.items():
            if brightness_values:
                plt.hist(
                    brightness_values,
                    bins=30,
                    alpha=0.7,
                    label=f"PCK {score} (n={len(brightness_values)})",
                )

        plt.title(f"Brightness Distribution by PCK Score\nDataset: {self.dataset_name}")
        plt.xlabel("Brightness Level")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_plot:
            plot_path = os.path.join(
                self.config.save_folder, f"brightness_histogram_{self.dataset_name}.png"
            )
            os.makedirs(self.config.save_folder, exist_ok=True)
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            print(f"✅ Histogram saved to: {plot_path}")
            plt.close()
        else:
            plt.show()

"""
PCK Brightness Distribution Analyzer.

This analyzer takes PCK scores from per-frame data, groups frames by their PCK scores,
extracts brightness for those frames, and analyzes the distribution.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
from core.base_classes import BaseAnalyzer
from extractors import MetricExtractorFactory
from processors import VideoPathResolver, FrameSynchronizer
from utils import ProgressTracker
import os


class PCKBrightnessAnalyzer(BaseAnalyzer):
    """Analyzer for PCK score vs brightness distribution."""

    def __init__(self, config, score_groups=None, bin_size=2):
        """Initialize analyzer with configuration."""
        super().__init__(config)
        self.path_resolver = VideoPathResolver(config)
        self.frame_sync = FrameSynchronizer(config)
        self.score_groups = score_groups  # List of specific PCK scores to analyze
        self.bin_size = bin_size  # Configurable bin size for brightness histogram

    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze PCK scores vs brightness distribution.

        Args:
            data: DataFrame containing per-frame PCK scores

        Returns:
            Dictionary containing brightness distributions for each PCK score
        """
        print("\n" + "=" * 60)
        print("Running PCK Brightness Distribution Analysis...")

        # Validate required columns
        required_columns = ["frame_idx"] + self.config.pck_per_frame_score_columns
        if not self.validate_data(data, required_columns):
            return {}

        results = {}

        # Process each PCK threshold column
        for pck_col in self.config.pck_per_frame_score_columns:
            print(f"\nProcessing {pck_col}...")
            pck_brightness_data = self._extract_pck_brightness_data(data, pck_col)

            if pck_brightness_data:
                results[pck_col] = self._analyze_pck_brightness_distribution(
                    pck_brightness_data, pck_col
                )
            else:
                print(f"No data extracted for {pck_col}")

        print("=" * 60)
        print("PCK Brightness Distribution Analysis completed.")

        return results

    def _extract_pck_brightness_data(
        self, data: pd.DataFrame, pck_column: str
    ) -> Dict[int, List[float]]:
        """
        Extract brightness data grouped by PCK scores.

        Args:
            data: DataFrame containing per-frame PCK scores
            pck_column: Name of the PCK score column to analyze

        Returns:
            Dictionary mapping PCK scores to lists of brightness values
        """
        pck_brightness_data = {}
        grouping_cols = self.config.get_grouping_columns()

        if not grouping_cols:
            print("Warning: No grouping columns found. Cannot perform analysis.")
            return pck_brightness_data

        grouped_data = data.groupby(grouping_cols)
        progress = ProgressTracker(
            len(grouped_data), f"Extracting brightness for {pck_column}"
        )

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
                print(f"Warning: Video not found for {video_info}. Skipping.")
                progress.update()
                continue

            # Extract brightness data for all frames
            brightness_extractor = MetricExtractorFactory.create_extractor(
                "brightness", video_path
            )
            brightness_data = brightness_extractor.extract()

            if not brightness_data:
                progress.update()
                continue

            # Get synchronization offset
            synced_start_frame = self.frame_sync.get_synced_start_frame(video_row_data)
            brightness_data_sliced = brightness_data[synced_start_frame:]

            # Group frames by their PCK scores
            for _, frame_row in group_data.iterrows():
                frame_idx = int(frame_row["frame_idx"])
                pck_score = frame_row[pck_column]

                # Round PCK score to nearest integer for grouping
                pck_score_int = int(round(pck_score))

                # Ensure frame index is within brightness data range
                if 0 <= frame_idx < len(brightness_data_sliced):
                    frame_brightness = brightness_data_sliced[frame_idx]

                    if pck_score_int not in pck_brightness_data:
                        pck_brightness_data[pck_score_int] = []

                    pck_brightness_data[pck_score_int].append(frame_brightness)

            progress.update()

        progress.finish()
        return pck_brightness_data

    def _analyze_pck_brightness_distribution(
        self, pck_brightness_data: Dict[int, List[float]], pck_column: str
    ) -> Dict[str, Any]:
        """
        Analyze brightness distribution for each PCK score.

        Args:
            pck_brightness_data: Dictionary mapping PCK scores to brightness values
            pck_column: Name of the PCK column being analyzed

        Returns:
            Dictionary containing analysis results
        """
        analysis_results = {
            "pck_column": pck_column,
            "pck_scores": [],
            "brightness_bins": [],
            "normalized_frequencies": [],
            "frame_counts": [],
            "brightness_stats": {},
            "raw_data": pck_brightness_data,
            "bin_size": self.bin_size,
        }

        # Find the extreme brightness values in the data
        all_brightness_values = []
        for brightness_list in pck_brightness_data.values():
            all_brightness_values.extend(brightness_list)

        if not all_brightness_values:
            print(f"Warning: No brightness values found for {pck_column}")
            return analysis_results

        min_brightness = min(all_brightness_values)
        max_brightness = max(all_brightness_values)

        # Extend range to extreme values beyond the data range
        # Add padding of 20% on each side to capture extreme values
        brightness_range = max_brightness - min_brightness
        padding = max(brightness_range * 0.2, 50)  # At least 50 units padding

        extended_min = max(0, min_brightness - padding)  # Don't go below 0
        extended_max = min(
            500, max_brightness + padding
        )  # Cap at 500 for extreme brightness

        # Create brightness bins using configurable bin size
        brightness_bins = np.arange(
            extended_min, extended_max + self.bin_size, self.bin_size
        )

        print(f"\nAnalyzing brightness distribution for {pck_column}:")
        print(f"Found {len(pck_brightness_data)} unique PCK scores")

        if not pck_brightness_data:
            print("No PCK brightness data available")
            return analysis_results

        print(f"Brightness range: {min_brightness:.1f} - {max_brightness:.1f}")
        print(f"Extended range: {extended_min:.1f} - {extended_max:.1f}")
        print(f"Bin size: {self.bin_size}")
        print(f"Number of bins: {len(brightness_bins) - 1}")
        print(f"Available PCK scores: {sorted(pck_brightness_data.keys())}")

        for pck_score in sorted(pck_brightness_data.keys()):
            brightness_values = pck_brightness_data[pck_score]
            frame_count = len(brightness_values)

            if frame_count == 0:
                continue

            # Calculate histogram
            hist, _ = np.histogram(brightness_values, bins=brightness_bins)

            # Normalize frequencies
            normalized_freq = hist / frame_count

            # Calculate brightness statistics
            brightness_stats = {
                "mean": np.mean(brightness_values),
                "std": np.std(brightness_values),
                "min": np.min(brightness_values),
                "max": np.max(brightness_values),
                "median": np.median(brightness_values),
            }

            # Store results
            analysis_results["pck_scores"].append(pck_score)
            analysis_results["brightness_bins"].append(
                brightness_bins[:-1]
            )  # Exclude last bin edge
            analysis_results["normalized_frequencies"].append(normalized_freq)
            analysis_results["frame_counts"].append(frame_count)
            analysis_results["brightness_stats"][pck_score] = brightness_stats

            print(
                f"  PCK {pck_score}: {frame_count} frames, "
                f"avg brightness: {brightness_stats['mean']:.1f}"
            )

        # Filter by score groups if specified
        if self.score_groups is not None:
            analysis_results = self._filter_by_score_groups(analysis_results)

        return analysis_results

    def _filter_by_score_groups(
        self, analysis_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Filter analysis results to include only specified PCK score groups.

        Args:
            analysis_results: Original analysis results

        Returns:
            Filtered analysis results containing only specified score groups
        """
        if not self.score_groups:
            return analysis_results

        # Convert score groups to integers for comparison
        target_scores = [int(score) for score in self.score_groups]

        filtered_results = {
            "pck_column": analysis_results["pck_column"],
            "pck_scores": [],
            "brightness_bins": [],
            "normalized_frequencies": [],
            "frame_counts": [],
            "brightness_stats": {},
            "raw_data": {},
            "bin_size": analysis_results.get("bin_size", self.bin_size),
        }

        print(f"Filtering to include only PCK scores: {target_scores}")

        # Filter each component
        for i, pck_score in enumerate(analysis_results["pck_scores"]):
            if pck_score in target_scores:
                filtered_results["pck_scores"].append(pck_score)
                filtered_results["brightness_bins"].append(
                    analysis_results["brightness_bins"][i]
                )
                filtered_results["normalized_frequencies"].append(
                    analysis_results["normalized_frequencies"][i]
                )
                filtered_results["frame_counts"].append(
                    analysis_results["frame_counts"][i]
                )
                filtered_results["brightness_stats"][pck_score] = analysis_results[
                    "brightness_stats"
                ][pck_score]
                filtered_results["raw_data"][pck_score] = analysis_results["raw_data"][
                    pck_score
                ]

        print(
            f"Filtered from {len(analysis_results['pck_scores'])} to {len(filtered_results['pck_scores'])} PCK scores"
        )

        return filtered_results

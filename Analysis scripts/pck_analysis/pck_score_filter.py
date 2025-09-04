"""
PCK Score Filter Script

Filters and displays specific PCK scores from the data.
Focus: PCK score filtering and inspection only.
"""

from processors.pck_data_loader import PCKDataLoader
import sys
import os
import pandas as pd
from typing import List, Optional, Dict

# Add the Analysis scripts directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class PCKScoreFilter:
    """Utility for filtering and inspecting PCK scores."""

    def __init__(self, dataset_name: str):
        """Initialize with dataset name."""
        self.dataset_name = dataset_name
        self.data_loader = PCKDataLoader(dataset_name)

    def get_unique_pck_scores(self, pck_threshold: str = None) -> Dict[str, List[int]]:
        """Get unique PCK scores for all or specific thresholds."""
        print(f"Finding unique PCK scores in {self.dataset_name}...")

        per_frame_data = self.data_loader.load_per_frame_data()
        if per_frame_data is None:
            print("❌ No per-frame data available")
            return {}

        from config import ConfigManager

        config = ConfigManager.load_config(self.dataset_name)

        unique_scores = {}

        thresholds_to_check = (
            [pck_threshold] if pck_threshold else config.pck_per_frame_score_columns
        )

        for threshold in thresholds_to_check:
            if threshold in per_frame_data.columns:
                # Round to nearest integer and get unique values
                scores = per_frame_data[threshold].round().astype(int).unique()
                scores = sorted(
                    [score for score in scores if not pd.isna(score)])
                unique_scores[threshold] = scores

                print(f"\n{threshold}:")
                print(f"  Range: {min(scores)} to {max(scores)}")
                print(f"  Count: {len(scores)} unique scores")
                print(f"  Scores: {scores}")

        return unique_scores

    def filter_by_score_range(
        self, min_score: int, max_score: int, pck_threshold: str = None
    ) -> Optional[pd.DataFrame]:
        """Filter data by PCK score range."""
        print(f"Filtering scores between {min_score} and {max_score}...")

        per_frame_data = self.data_loader.load_per_frame_data()
        if per_frame_data is None:
            return None

        from config import ConfigManager

        config = ConfigManager.load_config(self.dataset_name)

        if pck_threshold:
            thresholds = [pck_threshold]
        else:
            thresholds = config.pck_per_frame_score_columns

        filtered_data = per_frame_data.copy()

        for threshold in thresholds:
            if threshold in filtered_data.columns:
                # Filter by score range
                mask = (filtered_data[threshold] >= min_score) & (
                    filtered_data[threshold] <= max_score
                )
                filtered_data = filtered_data[mask]

        print(f"✅ Filtered data: {len(filtered_data)} frames")
        return filtered_data

    def filter_by_specific_scores(
        self, scores: List[int], pck_threshold: str = None
    ) -> Optional[pd.DataFrame]:
        """Filter data to include only specific PCK scores."""
        print(f"Filtering for specific scores: {scores}...")

        per_frame_data = self.data_loader.load_per_frame_data()
        if per_frame_data is None:
            return None

        from config import ConfigManager

        config = ConfigManager.load_config(self.dataset_name)

        if pck_threshold:
            thresholds = [pck_threshold]
        else:
            thresholds = config.pck_per_frame_score_columns

        # Create mask for any threshold containing the specified scores
        overall_mask = pd.Series(
            [False] * len(per_frame_data), index=per_frame_data.index
        )

        for threshold in thresholds:
            if threshold in per_frame_data.columns:
                # Round scores and check if they're in our target list
                rounded_scores = per_frame_data[threshold].round().astype(int)
                threshold_mask = rounded_scores.isin(scores)
                overall_mask = overall_mask | threshold_mask

        filtered_data = per_frame_data[overall_mask]
        print(f"✅ Filtered data: {len(filtered_data)} frames")
        return filtered_data

    def count_frames_per_score(
        self, pck_threshold: str = None
    ) -> Dict[str, Dict[int, int]]:
        """Count number of frames for each PCK score."""
        print("Counting frames per PCK score...")

        per_frame_data = self.data_loader.load_per_frame_data()
        if per_frame_data is None:
            return {}

        from config import ConfigManager

        config = ConfigManager.load_config(self.dataset_name)

        thresholds_to_check = (
            [pck_threshold] if pck_threshold else config.pck_per_frame_score_columns
        )

        frame_counts = {}

        for threshold in thresholds_to_check:
            if threshold in per_frame_data.columns:
                # Round scores and count
                rounded_scores = per_frame_data[threshold].round().astype(int)
                counts = rounded_scores.value_counts().sort_index()
                frame_counts[threshold] = counts.to_dict()

                print(f"\n{threshold} - Frame counts:")
                for score, count in sorted(counts.items()):
                    percentage = (count / len(per_frame_data)) * 100
                    print(
                        f"  Score {score}: {count} frames ({percentage:.1f}%)")

        return frame_counts

    def get_score_statistics(
        self, pck_threshold: str = None
    ) -> Dict[str, Dict[str, float]]:
        """Get statistical summary of PCK scores."""
        print("Calculating PCK score statistics...")

        per_frame_data = self.data_loader.load_per_frame_data()
        if per_frame_data is None:
            return {}

        from config import ConfigManager

        config = ConfigManager.load_config(self.dataset_name)

        thresholds_to_check = (
            [pck_threshold] if pck_threshold else config.pck_per_frame_score_columns
        )

        statistics = {}

        for threshold in thresholds_to_check:
            if threshold in per_frame_data.columns:
                scores = per_frame_data[threshold].dropna()

                stats = {
                    "mean": scores.mean(),
                    "median": scores.median(),
                    "std": scores.std(),
                    "min": scores.min(),
                    "max": scores.max(),
                    "q25": scores.quantile(0.25),
                    "q75": scores.quantile(0.75),
                }

                statistics[threshold] = stats

                print(f"\n{threshold} statistics:")
                print(f"  Mean: {stats['mean']:.2f}")
                print(f"  Median: {stats['median']:.2f}")
                print(f"  Std: {stats['std']:.2f}")
                print(f"  Range: {stats['min']:.1f} - {stats['max']:.1f}")

        return statistics

    def suggest_score_groups(self, num_groups: int = 3) -> Dict[str, List[List[int]]]:
        """Suggest PCK score groups for analysis."""
        print(f"Suggesting {num_groups} score groups for analysis...")

        unique_scores = self.get_unique_pck_scores()
        suggestions = {}

        for threshold, scores in unique_scores.items():
            if len(scores) < num_groups:
                suggestions[threshold] = [[score] for score in scores]
                continue

            # Create groups based on quantiles
            import numpy as np

            per_frame_data = self.data_loader.load_per_frame_data()
            threshold_data = per_frame_data[threshold].dropna()

            # Calculate quantile boundaries
            quantiles = np.linspace(0, 1, num_groups + 1)
            boundaries = [threshold_data.quantile(q) for q in quantiles]

            groups = []
            for i in range(num_groups):
                min_score = int(boundaries[i])
                max_score = int(boundaries[i + 1])

                # Get scores in this range
                group_scores = [
                    s for s in scores if min_score <= s <= max_score]
                if group_scores:
                    groups.append(group_scores)

            suggestions[threshold] = groups

            print(f"\n{threshold} - Suggested groups:")
            for i, group in enumerate(groups):
                print(
                    f"  Group {i + 1}: {group} (range: {min(group)}-{max(group)})")

        return suggestions


def main():
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description="PCK Score Filter")
    parser.add_argument(
        "dataset", help="Dataset name (e.g., 'movi', 'humaneva')")
    parser.add_argument(
        "--threshold", help="Specific PCK threshold to analyze")

    subparsers = parser.add_subparsers(
        dest="command", help="Available commands")

    # Unique scores command
    subparsers.add_parser("unique", help="Show unique PCK scores")

    # Filter range command
    range_parser = subparsers.add_parser("range", help="Filter by score range")
    range_parser.add_argument("min_score", type=int, help="Minimum PCK score")
    range_parser.add_argument("max_score", type=int, help="Maximum PCK score")

    # Filter specific scores command
    scores_parser = subparsers.add_parser(
        "scores", help="Filter by specific scores")
    scores_parser.add_argument(
        "score_list", nargs="+", type=int, help="List of PCK scores"
    )

    # Count frames command
    subparsers.add_parser("count", help="Count frames per score")

    # Statistics command
    subparsers.add_parser("stats", help="Show score statistics")

    # Suggest groups command
    suggest_parser = subparsers.add_parser(
        "suggest", help="Suggest score groups")
    suggest_parser.add_argument(
        "--groups", type=int, default=3, help="Number of groups"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    try:
        filter_tool = PCKScoreFilter(args.dataset)

        if args.command == "unique":
            filter_tool.get_unique_pck_scores(args.threshold)

        elif args.command == "range":
            filtered_data = filter_tool.filter_by_score_range(
                args.min_score, args.max_score, args.threshold
            )
            if filtered_data is not None:
                print(f"Filtered data shape: {filtered_data.shape}")

        elif args.command == "scores":
            filtered_data = filter_tool.filter_by_specific_scores(
                args.score_list, args.threshold
            )
            if filtered_data is not None:
                print(f"Filtered data shape: {filtered_data.shape}")

        elif args.command == "count":
            filter_tool.count_frames_per_score(args.threshold)

        elif args.command == "stats":
            filter_tool.get_score_statistics(args.threshold)

        elif args.command == "suggest":
            filter_tool.suggest_score_groups(args.groups)

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

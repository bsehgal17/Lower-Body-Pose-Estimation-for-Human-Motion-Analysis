"""
Simple Brightness Analyzer Script

Analyzes brightness distribution for specific PCK scores.
Focus: PCK brightness analysis only.
"""

import sys
import os
import pandas as pd
from typing import Dict, Any, Optional

# Add the Analysis scripts directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import ConfigManager
from analyzers import AnalyzerFactory
from simple_pck_loader import SimplePCKDataLoader


class SimpleBrightnessAnalyzer:
    """Simple brightness analyzer for PCK scores."""

    def __init__(self, dataset_name: str):
        """Initialize with dataset name."""
        self.dataset_name = dataset_name
        self.config = ConfigManager.load_config(dataset_name)
        self.data_loader = SimplePCKDataLoader(dataset_name)
        self.analyzer = AnalyzerFactory.create_analyzer("pck_brightness", self.config)

    def analyze_brightness_distribution(self) -> Optional[Dict[str, Any]]:
        """Run brightness distribution analysis."""
        print(f"Analyzing brightness distribution for {self.dataset_name}...")
        print("=" * 60)

        # Load per-frame data
        per_frame_data = self.data_loader.load_per_frame_data()
        if per_frame_data is None:
            print("❌ Cannot proceed without per-frame data")
            return None

        # Run analysis
        print("Running brightness analysis...")
        results = self.analyzer.analyze(per_frame_data)

        if results:
            print("✅ Analysis completed successfully")
            self._print_summary(results)
        else:
            print("❌ Analysis failed")

        return results

    def _print_summary(self, results: Dict[str, Any]):
        """Print analysis summary."""
        print("\n📊 Brightness Analysis Summary:")
        print("-" * 50)

        for pck_column, analysis in results.items():
            if "pck_scores" not in analysis:
                continue

            pck_scores = analysis["pck_scores"]
            frame_counts = analysis["frame_counts"]
            brightness_stats = analysis["brightness_stats"]

            print(f"\n{pck_column}:")
            print(f"  • PCK score range: {min(pck_scores)} to {max(pck_scores)}")
            print(f"  • Total frames: {sum(frame_counts)}")
            print(f"  • Number of unique PCK scores: {len(pck_scores)}")

            # Show top 3 PCK scores by frame count
            sorted_by_count = sorted(
                zip(pck_scores, frame_counts), key=lambda x: x[1], reverse=True
            )
            print("  • Top PCK scores by frame count:")
            for i, (pck, count) in enumerate(sorted_by_count[:3]):
                brightness_mean = brightness_stats[pck]["mean"]
                print(
                    f"    {i + 1}. PCK {pck}: {count} frames (avg brightness: {brightness_mean:.1f})"
                )

    def analyze_specific_pck_score(
        self, pck_score: int, pck_column: str = None
    ) -> Optional[Dict]:
        """Analyze brightness for a specific PCK score."""
        results = self.analyze_brightness_distribution()
        if not results:
            return None

        # If no column specified, use the first one
        if pck_column is None:
            pck_column = list(results.keys())[0]

        if pck_column not in results:
            print(f"❌ PCK column '{pck_column}' not found")
            return None

        analysis = results[pck_column]
        if pck_score not in analysis["brightness_stats"]:
            print(f"❌ PCK score {pck_score} not found in {pck_column}")
            return None

        # Extract data for specific PCK score
        pck_index = analysis["pck_scores"].index(pck_score)
        specific_data = {
            "pck_score": pck_score,
            "pck_column": pck_column,
            "frame_count": analysis["frame_counts"][pck_index],
            "brightness_stats": analysis["brightness_stats"][pck_score],
            "brightness_bins": analysis["brightness_bins"][pck_index],
            "normalized_frequencies": analysis["normalized_frequencies"][pck_index],
            "raw_brightness_values": analysis["raw_data"][pck_score],
        }

        print(f"\n🎯 Analysis for PCK Score {pck_score} ({pck_column}):")
        print("-" * 50)
        print(f"Frame count: {specific_data['frame_count']}")
        print(f"Brightness stats: {specific_data['brightness_stats']}")

        return specific_data

    def export_summary_to_csv(self, results: Dict[str, Any], filename: str = None):
        """Export summary statistics to CSV."""
        if not results:
            print("❌ No results to export")
            return

        if filename is None:
            filename = f"brightness_summary_{self.dataset_name}.csv"

        export_data = []
        for pck_column, analysis in results.items():
            if "pck_scores" not in analysis:
                continue

            for i, pck_score in enumerate(analysis["pck_scores"]):
                stats = analysis["brightness_stats"][pck_score]
                export_data.append(
                    {
                        "pck_threshold": pck_column,
                        "pck_score": pck_score,
                        "frame_count": analysis["frame_counts"][i],
                        "brightness_mean": stats["mean"],
                        "brightness_std": stats["std"],
                        "brightness_min": stats["min"],
                        "brightness_max": stats["max"],
                        "brightness_median": stats["median"],
                    }
                )

        df = pd.DataFrame(export_data)
        output_path = os.path.join(self.config.save_folder, filename)
        os.makedirs(self.config.save_folder, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"✅ Summary exported to: {output_path}")


def main():
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Simple Brightness Analyzer")
    parser.add_argument("dataset", help="Dataset name (e.g., 'movi', 'humaneva')")
    parser.add_argument("--pck-score", type=int, help="Analyze specific PCK score")
    parser.add_argument("--pck-column", help="Specific PCK column to analyze")
    parser.add_argument("--export", action="store_true", help="Export summary to CSV")

    args = parser.parse_args()

    try:
        analyzer = SimpleBrightnessAnalyzer(args.dataset)

        if args.pck_score is not None:
            # Analyze specific PCK score
            analyzer.analyze_specific_pck_score(args.pck_score, args.pck_column)
        else:
            # Run full analysis
            results = analyzer.analyze_brightness_distribution()

            if results and args.export:
                analyzer.export_summary_to_csv(results)

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

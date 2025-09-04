"""
Standalone script for PCK Brightness Distribution Analysis.

This script analyzes the relationship between PCK scores and brightness distribution
from per-frame data, creating line plots for each PCK score showing brightness
vs normalized frequency.

Usage:
    python pck_brightness_analysis.py [dataset_name]

Example:
    python pck_brightness_analysis.py movi
    python pck_brightness_analysis.py humaneva
"""

import sys
import os

# Add the Analysis scripts directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import ConfigManager
from analyzers import AnalyzerFactory
from visualizers import VisualizationFactory
from data_processor import DataProcessor
from utils import PerformanceMonitor
import argparse
from datetime import datetime


class PCKBrightnessAnalysisPipeline:
    """Pipeline for PCK brightness distribution analysis."""

    def __init__(self, dataset_name: str, score_groups: list = None):
        """Initialize the analysis pipeline."""
        self.dataset_name = dataset_name
        self.score_groups = score_groups
        self.config = ConfigManager.load_config(dataset_name)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create save folder
        os.makedirs(self.config.save_folder, exist_ok=True)

        # Initialize components
        self.data_processor = DataProcessor(self.config)

        # Create analyzer with score groups if specified
        if score_groups:
            self.analyzer = AnalyzerFactory.create_analyzer(
                "pck_brightness", self.config, score_groups=score_groups
            )
        else:
            self.analyzer = AnalyzerFactory.create_analyzer(
                "pck_brightness", self.config
            )

        self.visualizer = VisualizationFactory.create_visualizer(
            "pck_brightness", self.config
        )

    @PerformanceMonitor.timing_decorator
    def run_analysis(self, save_plots: bool = True):
        """Run the complete PCK brightness analysis."""
        print(
            f"Starting PCK Brightness Analysis for {self.dataset_name.upper()} dataset..."
        )
        print(f"Timestamp: {self.timestamp}")
        print(f"Save folder: {self.config.save_folder}")
        print("=" * 80)

        # Step 1: Load per-frame PCK data
        print("Step 1: Loading per-frame PCK data...")
        per_frame_data = self.data_processor.load_pck_per_frame_scores()

        if per_frame_data is None or per_frame_data.empty:
            print("❌ Error: No per-frame PCK data found or loaded successfully.")
            return None

        print(f"✅ Loaded {len(per_frame_data)} per-frame records")
        print(f"   Available PCK columns: {self.config.pck_per_frame_score_columns}")
        print()

        # Step 2: Run PCK brightness analysis
        print("Step 2: Running PCK brightness distribution analysis...")
        analysis_results = self.analyzer.analyze(per_frame_data)

        if not analysis_results:
            print("❌ Error: No analysis results generated.")
            return None

        print(f"✅ Analysis completed for {len(analysis_results)} PCK threshold(s)")

        # Print summary statistics
        self._print_analysis_summary(analysis_results)
        print()

        # Step 3: Create visualizations
        if save_plots:
            print("Step 3: Creating visualizations...")
            save_path = f"pck_brightness_{self.dataset_name}_{self.timestamp}"

            # Create individual plots for each PCK threshold
            self.visualizer.create_plot(analysis_results, save_path)

            # Create combined summary plot
            self.visualizer.create_combined_summary_plot(analysis_results, save_path)

            print("✅ All visualizations created successfully")
        else:
            print("Step 3: Skipping visualizations (save_plots=False)")

        print("\n" + "=" * 80)
        print(
            f"PCK Brightness Analysis completed for {self.dataset_name.upper()} dataset!"
        )

        return analysis_results

    def _print_analysis_summary(self, analysis_results):
        """Print summary of analysis results."""
        print("\n📊 Analysis Summary:")
        print("-" * 40)

        for pck_column, results in analysis_results.items():
            if not results or "pck_scores" not in results:
                continue

            pck_scores = results["pck_scores"]
            frame_counts = results["frame_counts"]
            brightness_stats = results["brightness_stats"]

            total_frames = sum(frame_counts)
            unique_pck_scores = len(pck_scores)

            print(f"\n{pck_column}:")
            print(f"  • Unique PCK scores: {unique_pck_scores}")
            print(f"  • Total frames analyzed: {total_frames}")
            print(f"  • PCK score range: {min(pck_scores)} to {max(pck_scores)}")

            # Find PCK score with most frames
            max_frames_idx = frame_counts.index(max(frame_counts))
            most_common_pck = pck_scores[max_frames_idx]
            most_common_count = frame_counts[max_frames_idx]

            print(
                f"  • Most common PCK score: {most_common_pck} ({most_common_count} frames)"
            )

            # Calculate average brightness across all PCK scores
            all_brightness_means = [brightness_stats[pck]["mean"] for pck in pck_scores]
            avg_brightness = sum(all_brightness_means) / len(all_brightness_means)
            print(f"  • Average brightness across all PCK scores: {avg_brightness:.1f}")

    def export_results_to_csv(self, analysis_results, filename: str = None):
        """Export analysis results to CSV for further analysis."""
        import pandas as pd

        if filename is None:
            filename = (
                f"pck_brightness_results_{self.dataset_name}_{self.timestamp}.csv"
            )

        export_data = []

        for pck_column, results in analysis_results.items():
            if not results or "pck_scores" not in results:
                continue

            pck_scores = results["pck_scores"]
            frame_counts = results["frame_counts"]
            brightness_stats = results["brightness_stats"]

            for pck_score, frame_count in zip(pck_scores, frame_counts):
                stats = brightness_stats[pck_score]
                export_data.append(
                    {
                        "pck_threshold": pck_column,
                        "pck_score": pck_score,
                        "frame_count": frame_count,
                        "brightness_mean": stats["mean"],
                        "brightness_std": stats["std"],
                        "brightness_min": stats["min"],
                        "brightness_max": stats["max"],
                        "brightness_median": stats["median"],
                    }
                )

        if export_data:
            df = pd.DataFrame(export_data)
            csv_path = os.path.join(self.config.save_folder, filename)
            df.to_csv(csv_path, index=False)
            print(f"✅ Results exported to: {csv_path}")
            return csv_path
        else:
            print("❌ No data to export")
            return None


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="PCK Brightness Distribution Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python pck_brightness_analysis.py movi
    python pck_brightness_analysis.py humaneva --no-plots
    python pck_brightness_analysis.py movi --export-csv
    python pck_brightness_analysis.py movi --scores 85 90 95
    python pck_brightness_analysis.py humaneva --scores 80 85 90 --no-plots
    python pck_brightness_analysis.py movi --demo
    python pck_brightness_analysis.py humaneva --demo --export-csv
        """,
    )

    parser.add_argument("dataset", help="Dataset name (e.g., 'movi', 'humaneva')")

    parser.add_argument(
        "--scores",
        nargs="+",
        type=int,
        help="Specific PCK scores to analyze (e.g., --scores 85 90 95)",
    )

    parser.add_argument(
        "--no-plots", action="store_true", help="Skip creating plots (analysis only)"
    )

    parser.add_argument(
        "--export-csv", action="store_true", help="Export results to CSV file"
    )

    parser.add_argument(
        "--demo", action="store_true", help="Run demo with different score groups"
    )

    args = parser.parse_args()

    try:
        if args.demo:
            # Demo mode: Run analysis with different score groups
            print("🚀 Running PCK Brightness Analysis Demo")
            print("=" * 60)

            # Analysis 1: All scores (no filtering)
            print("\n📊 Analysis 1: All PCK scores")
            print("-" * 40)
            pipeline_all = PCKBrightnessAnalysisPipeline(
                args.dataset, score_groups=None
            )
            results_all = pipeline_all.run_analysis(save_plots=not args.no_plots)

            if results_all and args.export_csv:
                pipeline_all.export_results_to_csv(
                    results_all, "all_scores_analysis.csv"
                )

            # Analysis 2: Only scores 0 and 100
            print("\n📊 Analysis 2: PCK scores 0 and 100 only")
            print("-" * 40)
            pipeline_filtered = PCKBrightnessAnalysisPipeline(
                args.dataset, score_groups=[0, 100]
            )
            results_filtered = pipeline_filtered.run_analysis(
                save_plots=not args.no_plots
            )

            if results_filtered and args.export_csv:
                pipeline_filtered.export_results_to_csv(
                    results_filtered, "scores_0_100_analysis.csv"
                )

            print("\n" + "=" * 60)
            print("🎯 Demo Summary:")
            print(
                f"   Analysis 1 (All scores): {'✅ Completed' if results_all else '❌ Failed'}"
            )
            print(
                f"   Analysis 2 (Scores 0,100): {'✅ Completed' if results_filtered else '❌ Failed'}"
            )
            print("=" * 60)

        else:
            # Regular mode: Use command line arguments
            pipeline = PCKBrightnessAnalysisPipeline(args.dataset, args.scores)
            results = pipeline.run_analysis(save_plots=not args.no_plots)

            if results and args.export_csv:
                pipeline.export_results_to_csv(results)

        print("\n✅ Script completed successfully!")

    except FileNotFoundError as e:
        print(f"❌ Error: Required file not found - {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"❌ Error: Configuration issue - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error occurred: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

"""
Simple Statistical Analysis Script

Runs statistical analyses (ANOVA, bin analysis) on PCK and brightness data.
Focus: Statistical testing only.
"""

from core.data_processor import DataProcessor
from analyzers import AnalyzerFactory
from config import ConfigManager
import sys
import os
import pandas as pd
from typing import Optional, Dict, Any

# Add the Analysis scripts directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class SimpleStatisticalAnalyzer:
    """Simple statistical analyzer for PCK data."""

    def __init__(self, dataset_name: str):
        """Initialize with dataset name."""
        self.dataset_name = dataset_name
        self.config = ConfigManager.load_config(dataset_name)
        self.data_processor = DataProcessor(self.config)

    def run_anova_analysis(self) -> Optional[Dict[str, Any]]:
        """Run ANOVA analysis on per-frame data."""
        print(f"Running ANOVA analysis for {self.dataset_name}...")
        print("=" * 50)

        # Load per-frame data
        per_frame_data = self.data_processor.load_pck_per_frame_scores()
        if per_frame_data is None:
            print("❌ Cannot load per-frame data")
            return None

        try:
            # Create ANOVA analyzer
            anova_analyzer = AnalyzerFactory.create_analyzer(
                "anova", self.config)

            # Run analysis for brightness metric
            print("Running ANOVA for brightness metric...")
            results = anova_analyzer.analyze(per_frame_data, "brightness")

            if results:
                print("✅ ANOVA analysis completed")
                self._print_anova_summary(results)
            else:
                print("❌ ANOVA analysis failed")

            return results

        except Exception as e:
            print(f"❌ Error in ANOVA analysis: {e}")
            return None

    def run_bin_analysis(self) -> Optional[Dict[str, Any]]:
        """Run bin analysis on per-frame data."""
        print(f"Running bin analysis for {self.dataset_name}...")
        print("=" * 50)

        # Load per-frame data
        per_frame_data = self.data_processor.load_pck_per_frame_scores()
        if per_frame_data is None:
            print("❌ Cannot load per-frame data")
            return None

        try:
            # Create bin analyzer
            bin_analyzer = AnalyzerFactory.create_analyzer(
                "bin_analysis", self.config)

            # Run analysis for brightness metric
            print("Running bin analysis for brightness metric...")
            results = bin_analyzer.analyze(per_frame_data, "brightness")

            if results:
                print("✅ Bin analysis completed")
                self._print_bin_summary(results)
            else:
                print("❌ Bin analysis failed")

            return results

        except Exception as e:
            print(f"❌ Error in bin analysis: {e}")
            return None

    def run_all_statistical_analyses(self) -> Dict[str, Any]:
        """Run all available statistical analyses."""
        print(f"Running all statistical analyses for {self.dataset_name}...")
        print("=" * 60)

        all_results = {}

        # Run ANOVA
        anova_results = self.run_anova_analysis()
        if anova_results:
            all_results["anova"] = anova_results

        print("\n" + "-" * 60 + "\n")

        # Run bin analysis
        bin_results = self.run_bin_analysis()
        if bin_results:
            all_results["bin_analysis"] = bin_results

        print("\n" + "=" * 60)
        print("Statistical Analysis Summary:")
        print(
            f"  • ANOVA: {'✅ Completed' if 'anova' in all_results else '❌ Failed'}")
        print(
            f"  • Bin Analysis: {'✅ Completed' if 'bin_analysis' in all_results else '❌ Failed'}"
        )

        return all_results

    def _print_anova_summary(self, results: Dict[str, Any]):
        """Print ANOVA analysis summary."""
        print("\n📊 ANOVA Analysis Summary:")
        print("-" * 40)

        if "p_values" in results:
            for pck_col, p_value in results["p_values"].items():
                significance = "significant" if p_value < 0.05 else "not significant"
                print(f"  {pck_col}: p-value = {p_value:.6f} ({significance})")

        if "f_statistics" in results:
            print("\n  F-statistics:")
            for pck_col, f_stat in results["f_statistics"].items():
                print(f"    {pck_col}: F = {f_stat:.4f}")

    def _print_bin_summary(self, results: Dict[str, Any]):
        """Print bin analysis summary."""
        print("\n📊 Bin Analysis Summary:")
        print("-" * 40)

        if "chi2_stats" in results:
            for pck_col, chi2_stat in results["chi2_stats"].items():
                print(f"  {pck_col}: Chi-square = {chi2_stat:.4f}")

        if "p_values" in results:
            print("\n  P-values:")
            for pck_col, p_value in results["p_values"].items():
                significance = "significant" if p_value < 0.05 else "not significant"
                print(f"    {pck_col}: p = {p_value:.6f} ({significance})")

    def export_statistical_results(self, results: Dict[str, Any], filename: str = None):
        """Export statistical results to CSV."""
        if not results:
            print("❌ No results to export")
            return

        if filename is None:
            filename = f"statistical_results_{self.dataset_name}.csv"

        export_data = []

        for analysis_type, analysis_results in results.items():
            if "p_values" not in analysis_results:
                continue

            for pck_col, p_value in analysis_results["p_values"].items():
                row = {
                    "analysis_type": analysis_type,
                    "pck_threshold": pck_col,
                    "p_value": p_value,
                    "significant": p_value < 0.05,
                }

                # Add additional statistics if available
                if (
                    "f_statistics" in analysis_results
                    and pck_col in analysis_results["f_statistics"]
                ):
                    row["f_statistic"] = analysis_results["f_statistics"][pck_col]

                if (
                    "chi2_stats" in analysis_results
                    and pck_col in analysis_results["chi2_stats"]
                ):
                    row["chi2_statistic"] = analysis_results["chi2_stats"][pck_col]

                export_data.append(row)

        if export_data:
            df = pd.DataFrame(export_data)
            output_path = os.path.join(self.config.save_folder, filename)
            os.makedirs(self.config.save_folder, exist_ok=True)
            df.to_csv(output_path, index=False)
            print(f"✅ Statistical results exported to: {output_path}")
        else:
            print("❌ No statistical data to export")

    def compare_pck_thresholds(self, results: Dict[str, Any]):
        """Compare results across different PCK thresholds."""
        if not results:
            return

        print("\n🔍 PCK Threshold Comparison:")
        print("-" * 50)

        for analysis_type, analysis_results in results.items():
            if "p_values" not in analysis_results:
                continue

            print(f"\n{analysis_type.upper()}:")

            # Sort by p-value
            sorted_results = sorted(
                analysis_results["p_values"].items(), key=lambda x: x[1]
            )

            for pck_col, p_value in sorted_results:
                significance = (
                    "***"
                    if p_value < 0.001
                    else "**"
                    if p_value < 0.01
                    else "*"
                    if p_value < 0.05
                    else "ns"
                )
                print(f"  {pck_col}: p = {p_value:.6f} {significance}")


def main():
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Simple Statistical Analyzer")
    parser.add_argument(
        "dataset", help="Dataset name (e.g., 'movi', 'humaneva')")
    parser.add_argument(
        "--type",
        choices=["anova", "bin", "all"],
        default="all",
        help="Type of statistical analysis",
    )
    parser.add_argument("--export", action="store_true",
                        help="Export results to CSV")
    parser.add_argument(
        "--compare", action="store_true", help="Compare results across PCK thresholds"
    )

    args = parser.parse_args()

    try:
        analyzer = SimpleStatisticalAnalyzer(args.dataset)

        # Run analyses based on type
        if args.type == "anova":
            results = {"anova": analyzer.run_anova_analysis()}
        elif args.type == "bin":
            results = {"bin_analysis": analyzer.run_bin_analysis()}
        else:  # all
            results = analyzer.run_all_statistical_analyses()

        # Export if requested
        if args.export and results:
            analyzer.export_statistical_results(results)

        # Compare thresholds if requested
        if args.compare and results:
            analyzer.compare_pck_thresholds(results)

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

"""
Master Orchestrator Script

Coordinates and orchestrates all PCK brightness analysis components.
Focus: Workflow coordination and high-level automation.
"""

import sys
import os
import time
from typing import List

# Add the Analysis scripts directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pck_score_filter import PCKScoreFilter
from brightness_extractor import BrightnessExtractor
from distribution_calculator import DistributionCalculator
from line_plot_creator import LinePlotCreator
from statistical_summary_generator import StatisticalSummaryGenerator


class MasterOrchestrator:
    """Master orchestrator for PCK brightness analysis workflow."""

    def __init__(self, dataset_name: str):
        """Initialize with dataset name."""
        self.dataset_name = dataset_name

        # Initialize all components
        self.score_filter = PCKScoreFilter(dataset_name)
        self.brightness_extractor = BrightnessExtractor(dataset_name)
        self.distribution_calculator = DistributionCalculator(dataset_name)
        self.plot_creator = LinePlotCreator(dataset_name)
        self.stats_generator = StatisticalSummaryGenerator(dataset_name)

        print(f"🎭 Master Orchestrator initialized for dataset: {dataset_name}")

    def run_complete_analysis(
        self,
        target_scores: List[int],
        pck_threshold: str = None,
        score_groups: List[int] = None,
        bin_size: int = 5,
        create_plots: bool = True,
        generate_stats: bool = True,
        export_excel: bool = True,
        output_prefix: str = "complete_analysis",
    ) -> dict:
        """Run complete PCK brightness analysis workflow."""
        print(f"\n🚀 Starting complete analysis for PCK scores: {target_scores}")
        print("=" * 70)

        start_time = time.time()
        results = {
            "dataset": self.dataset_name,
            "target_scores": target_scores,
            "pck_threshold": pck_threshold,
            "score_groups": score_groups,
            "bin_size": bin_size,
            "files_created": [],
            "analysis_summary": {},
            "errors": [],
        }

        try:
            # Step 1: Filter and inspect PCK scores
            print("\n📋 Step 1: Filtering PCK scores...")
            filtered_data = self.score_filter.filter_and_inspect_scores(
                target_scores, pck_threshold, score_groups
            )

            if filtered_data.empty:
                results["errors"].append("No data available after filtering")
                return results

            results["analysis_summary"]["filtered_frames"] = len(filtered_data)

            # Step 2: Extract brightness data
            print("\n🔆 Step 2: Extracting brightness data...")
            brightness_data = self.brightness_extractor.extract_brightness_for_scores(
                target_scores, pck_threshold, score_groups
            )

            if not brightness_data:
                results["errors"].append("No brightness data extracted")
                return results

            results["analysis_summary"]["brightness_extractions"] = len(brightness_data)

            # Step 3: Calculate distributions
            print("\n📊 Step 3: Calculating distributions...")
            distributions = (
                self.distribution_calculator.calculate_distributions_for_scores(
                    target_scores, pck_threshold, bin_size
                )
            )

            if not distributions:
                results["errors"].append("No distributions calculated")
                return results

            results["analysis_summary"]["distributions_calculated"] = len(distributions)

            # Step 4: Create visualizations
            if create_plots:
                print("\n📈 Step 4: Creating visualizations...")

                # Frequency line plot
                plot_path = self.plot_creator.create_frequency_line_plot(
                    target_scores,
                    pck_threshold,
                    bin_size,
                    True,
                    f"{output_prefix}_frequency_plot.svg",
                )
                if plot_path:
                    results["files_created"].append(plot_path)

                # Smoothed line plot
                smoothed_path = self.plot_creator.create_smoothed_line_plot(
                    target_scores,
                    pck_threshold,
                    bin_size,
                    0.3,
                    True,
                    f"{output_prefix}_smoothed_plot.svg",
                )
                if smoothed_path:
                    results["files_created"].append(smoothed_path)

                # Overlay plot with first score highlighted
                if len(target_scores) > 1:
                    overlay_path = self.plot_creator.create_overlay_plot(
                        target_scores,
                        pck_threshold,
                        bin_size,
                        target_scores[0],
                        True,
                        f"{output_prefix}_overlay_plot.svg",
                    )
                    if overlay_path:
                        results["files_created"].append(overlay_path)

            # Step 5: Generate statistical summaries
            if generate_stats:
                print("\n📋 Step 5: Generating statistical summaries...")

                # Comprehensive report
                report, report_path = self.stats_generator.generate_summary_report(
                    target_scores,
                    pck_threshold,
                    bin_size,
                    True,
                    f"{output_prefix}_summary_report.json",
                )
                if report_path:
                    results["files_created"].append(report_path)
                    results["analysis_summary"]["report_sections"] = len(report)

                # Excel export
                if export_excel:
                    excel_path = self.stats_generator.export_all_statistics_to_excel(
                        target_scores,
                        pck_threshold,
                        bin_size,
                        f"{output_prefix}_all_statistics.xlsx",
                    )
                    if excel_path:
                        results["files_created"].append(excel_path)

            # Final summary
            end_time = time.time()
            results["analysis_summary"]["execution_time_seconds"] = round(
                end_time - start_time, 2
            )
            results["analysis_summary"]["files_created_count"] = len(
                results["files_created"]
            )

            print("\n✅ Complete analysis finished successfully!")
            print(
                f"⏱️  Execution time: {results['analysis_summary']['execution_time_seconds']} seconds"
            )
            print(
                f"📁 Files created: {results['analysis_summary']['files_created_count']}"
            )

        except Exception as e:
            error_msg = f"Analysis failed: {str(e)}"
            results["errors"].append(error_msg)
            print(f"❌ {error_msg}")
            import traceback

            traceback.print_exc()

        return results

    def run_quick_analysis(
        self, target_scores: List[int], pck_threshold: str = None, bin_size: int = 5
    ) -> dict:
        """Run quick analysis with basic outputs."""
        print(f"\n⚡ Starting quick analysis for PCK scores: {target_scores}")

        start_time = time.time()
        results = {
            "dataset": self.dataset_name,
            "target_scores": target_scores,
            "quick_stats": {},
            "plot_created": False,
        }

        try:
            # Get basic statistics
            desc_stats = self.stats_generator.generate_descriptive_statistics(
                target_scores, pck_threshold, bin_size
            )

            if not desc_stats.empty:
                results["quick_stats"] = {
                    "total_scores": len(desc_stats),
                    "total_frames": desc_stats["Total_Frames"].sum(),
                    "mean_brightness_range": {
                        "min": desc_stats["Mean_Brightness"].min(),
                        "max": desc_stats["Mean_Brightness"].max(),
                    },
                    "most_frames_score": int(
                        desc_stats.loc[desc_stats["Total_Frames"].idxmax(), "PCK_Score"]
                    ),
                    "highest_brightness_score": int(
                        desc_stats.loc[
                            desc_stats["Mean_Brightness"].idxmax(), "PCK_Score"
                        ]
                    ),
                }

            # Create one basic plot
            plot_path = self.plot_creator.create_frequency_line_plot(
                target_scores,
                pck_threshold,
                bin_size,
                True,
                f"quick_analysis_{self.dataset_name}.svg",
            )
            results["plot_created"] = bool(plot_path)

            end_time = time.time()
            results["execution_time"] = round(end_time - start_time, 2)

            print(f"✅ Quick analysis completed in {results['execution_time']} seconds")

        except Exception as e:
            print(f"❌ Quick analysis failed: {e}")
            results["error"] = str(e)

        return results

    def run_comparison_analysis(
        self,
        score_groups: List[List[int]],
        group_labels: List[str] = None,
        pck_threshold: str = None,
        bin_size: int = 5,
    ) -> dict:
        """Run comparison analysis between score groups."""
        print(f"\n🔍 Starting comparison analysis for {len(score_groups)} score groups")

        start_time = time.time()
        results = {
            "dataset": self.dataset_name,
            "score_groups": score_groups,
            "group_labels": group_labels
            or [f"Group {i + 1}" for i in range(len(score_groups))],
            "files_created": [],
            "group_summaries": [],
        }

        try:
            # Analyze each group
            for i, (scores, label) in enumerate(
                zip(score_groups, results["group_labels"])
            ):
                print(f"\n   Analyzing {label}: {scores}")

                # Get statistics for this group
                desc_stats = self.stats_generator.generate_descriptive_statistics(
                    scores, pck_threshold, bin_size
                )

                if not desc_stats.empty:
                    group_summary = {
                        "label": label,
                        "scores": scores,
                        "total_frames": desc_stats["Total_Frames"].sum(),
                        "mean_brightness": desc_stats["Mean_Brightness"].mean(),
                        "brightness_variability": desc_stats["Std_Brightness"].mean(),
                    }
                    results["group_summaries"].append(group_summary)

            # Create comparison plot
            comparison_path = self.plot_creator.create_comparison_plot(
                score_groups,
                results["group_labels"],
                pck_threshold,
                bin_size,
                True,
                f"comparison_analysis_{self.dataset_name}.svg",
            )
            if comparison_path:
                results["files_created"].append(comparison_path)

            # Generate comparison statistics if multiple groups
            if len(score_groups) > 1:
                all_scores = [score for group in score_groups for score in group]
                comp_stats = self.stats_generator.generate_distribution_comparison(
                    all_scores, pck_threshold, bin_size
                )

                if not comp_stats.empty:
                    # Save comparison stats
                    from config import ConfigManager

                    config = ConfigManager.load_config(self.dataset_name)

                    comp_path = os.path.join(
                        config.save_folder, f"comparison_stats_{self.dataset_name}.csv"
                    )
                    comp_stats.to_csv(comp_path, index=False)
                    results["files_created"].append(comp_path)

            end_time = time.time()
            results["execution_time"] = round(end_time - start_time, 2)

            print(
                f"✅ Comparison analysis completed in {results['execution_time']} seconds"
            )

        except Exception as e:
            print(f"❌ Comparison analysis failed: {e}")
            results["error"] = str(e)

        return results

    def display_workflow_status(self):
        """Display the status of all workflow components."""
        print(f"\n🎭 WORKFLOW STATUS - Dataset: {self.dataset_name}")
        print("=" * 60)

        components = [
            ("📋 PCK Score Filter", self.score_filter),
            ("🔆 Brightness Extractor", self.brightness_extractor),
            ("📊 Distribution Calculator", self.distribution_calculator),
            ("📈 Line Plot Creator", self.plot_creator),
            ("📋 Statistical Summary Generator", self.stats_generator),
        ]

        for name, component in components:
            status = "✅ Ready" if component else "❌ Not Available"
            print(f"{name}: {status}")

        print(f"\n🗂️  Dataset Configuration: {self.dataset_name}")

        try:
            from config import ConfigManager

            config = ConfigManager.load_config(self.dataset_name)
            print(f"📁 Save Folder: {config.save_folder}")
            print(
                f"📹 Video Folder: {getattr(config, 'video_folder', 'Not specified')}"
            )
        except Exception as e:
            print(f"⚠️  Configuration issue: {e}")


def main():
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Master Orchestrator for PCK Brightness Analysis"
    )
    parser.add_argument("dataset", help="Dataset name (e.g., 'movi', 'humaneva')")
    parser.add_argument("scores", nargs="+", type=int, help="PCK scores to analyze")
    parser.add_argument("--threshold", help="Specific PCK threshold to analyze")
    parser.add_argument(
        "--score-groups", nargs="+", type=int, help="Score groups filter"
    )
    parser.add_argument("--bin-size", type=int, default=5, help="Brightness bin size")
    parser.add_argument(
        "--analysis-type",
        choices=["complete", "quick", "comparison"],
        default="complete",
        help="Type of analysis to run",
    )
    parser.add_argument(
        "--output-prefix", default="analysis", help="Prefix for output files"
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip plot creation")
    parser.add_argument(
        "--no-stats", action="store_true", help="Skip statistical analysis"
    )
    parser.add_argument("--no-excel", action="store_true", help="Skip Excel export")
    parser.add_argument(
        "--status", action="store_true", help="Show workflow status and exit"
    )

    args = parser.parse_args()

    try:
        orchestrator = MasterOrchestrator(args.dataset)

        if args.status:
            orchestrator.display_workflow_status()
            return

        if args.analysis_type == "complete":
            results = orchestrator.run_complete_analysis(
                target_scores=args.scores,
                pck_threshold=args.threshold,
                score_groups=args.score_groups,
                bin_size=args.bin_size,
                create_plots=not args.no_plots,
                generate_stats=not args.no_stats,
                export_excel=not args.no_excel,
                output_prefix=args.output_prefix,
            )

            print("\n📊 FINAL RESULTS:")
            print(f"   Files created: {len(results['files_created'])}")
            if results["errors"]:
                print(f"   Errors: {len(results['errors'])}")
                for error in results["errors"]:
                    print(f"     - {error}")

        elif args.analysis_type == "quick":
            results = orchestrator.run_quick_analysis(
                target_scores=args.scores,
                pck_threshold=args.threshold,
                bin_size=args.bin_size,
            )

            print("\n📊 QUICK RESULTS:")
            if "quick_stats" in results:
                for key, value in results["quick_stats"].items():
                    print(f"   {key}: {value}")

        elif args.analysis_type == "comparison":
            # For comparison, split scores into groups (example: every 2 scores)
            score_groups = [
                args.scores[i : i + 2] for i in range(0, len(args.scores), 2)
            ]

            results = orchestrator.run_comparison_analysis(
                score_groups=score_groups,
                pck_threshold=args.threshold,
                bin_size=args.bin_size,
            )

            print("\n📊 COMPARISON RESULTS:")
            print(f"   Groups analyzed: {len(results['group_summaries'])}")
            for summary in results["group_summaries"]:
                print(
                    f"   {summary['label']}: {summary['total_frames']} frames, "
                    f"avg brightness: {summary['mean_brightness']:.2f}"
                )

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

"""
Master Ground Truth Analysis Orchestrator Script

Coordinates and orchestrates all ground truth PCK-brightness analysis components.
Focus: Workflow coordination for complete GT analysis.
"""

import sys
import os
import time
from typing import List

# Add the Analysis scripts directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gt_data_loader import GroundTruthDataLoader
from joint_brightness_extractor import JointBrightnessExtractor
from gt_pck_brightness_analyzer import GTPCKBrightnessAnalyzer
from gt_distribution_calculator import GTDistributionCalculator
from gt_visualization_creator import GTVisualizationCreator


class MasterGTAnalysisOrchestrator:
    """Master orchestrator for ground truth PCK-brightness analysis workflow."""

    def __init__(self, dataset_name: str):
        """Initialize with dataset name."""
        self.dataset_name = dataset_name

        # Initialize all GT analysis components
        self.gt_loader = GroundTruthDataLoader(dataset_name)
        self.brightness_extractor = JointBrightnessExtractor(dataset_name)
        self.pck_analyzer = GTPCKBrightnessAnalyzer(dataset_name)
        self.distribution_calculator = GTDistributionCalculator(dataset_name)
        self.visualization_creator = GTVisualizationCreator(dataset_name)

        print(
            f"🎭 Master GT Analysis Orchestrator initialized for dataset: {dataset_name}"
        )

    def run_complete_gt_analysis(
        self,
        joint_names: List[str] = None,
        pck_threshold: str = None,
        video_path: str = None,
        target_pck_scores: List[int] = None,
        bin_size: int = 5,
        sampling_radius: int = 3,
        create_visualizations: bool = True,
        export_data: bool = True,
        output_prefix: str = "gt_analysis",
        **gt_filter_kwargs,
    ) -> dict:
        """Run complete ground truth PCK-brightness analysis workflow."""
        print(f"\n🚀 Starting complete GT analysis for joints: {joint_names}")
        print("=" * 80)

        start_time = time.time()
        results = {
            "dataset": self.dataset_name,
            "joint_names": joint_names,
            "pck_threshold": pck_threshold,
            "target_pck_scores": target_pck_scores,
            "parameters": {
                "bin_size": bin_size,
                "sampling_radius": sampling_radius,
                "gt_filters": gt_filter_kwargs,
            },
            "files_created": [],
            "analysis_summary": {},
            "errors": [],
        }

        try:
            # Step 1: Load and inspect ground truth data
            print("\n📋 Step 1: Loading ground truth data...")
            gt_summary = self.gt_loader.get_data_summary()

            if not gt_summary:
                results["errors"].append("Failed to load ground truth data")
                return results

            results["analysis_summary"]["gt_records"] = gt_summary.get(
                "total_records", 0
            )
            results["analysis_summary"]["available_joints"] = len(
                gt_summary.get("available_joints", [])
            )

            # Step 2: Extract joint coordinates
            print("\n📍 Step 2: Extracting joint coordinates...")
            joint_coordinates = self.gt_loader.extract_joint_coordinates(
                joint_names, **gt_filter_kwargs
            )

            if not joint_coordinates:
                results["errors"].append("No joint coordinates extracted")
                return results

            results["analysis_summary"]["joints_extracted"] = len(joint_coordinates)

            # Step 3: Extract brightness at joint coordinates
            print("\n🔆 Step 3: Extracting brightness at joint coordinates...")
            brightness_data = self.brightness_extractor.extract_brightness_for_joints(
                joint_names=joint_names,
                video_path=video_path,
                sampling_radius=sampling_radius,
                **gt_filter_kwargs,
            )

            if not brightness_data:
                results["errors"].append("No brightness data extracted")
                return results

            results["analysis_summary"]["brightness_extractions"] = len(brightness_data)

            # Step 4: Analyze PCK-brightness correlations
            print("\n📊 Step 4: Analyzing PCK-brightness correlations...")
            correlation_results = self.pck_analyzer.analyze_pck_brightness_correlation(
                joint_names=joint_names,
                pck_threshold=pck_threshold,
                video_path=video_path,
                sampling_radius=sampling_radius,
                **gt_filter_kwargs,
            )

            if not correlation_results.get("correlations"):
                results["errors"].append("No correlation analysis completed")
                return results

            results["analysis_summary"]["correlations_calculated"] = len(
                correlation_results["correlations"]
            )

            # Step 5: Calculate distributions
            print("\n📈 Step 5: Calculating distributions...")
            distribution_results = (
                self.distribution_calculator.run_complete_distribution_analysis(
                    joint_names=joint_names,
                    pck_threshold=pck_threshold,
                    bin_size=bin_size,
                    video_path=video_path,
                    **gt_filter_kwargs,
                )
            )

            if not distribution_results:
                results["errors"].append("No distribution analysis completed")
                return results

            results["analysis_summary"]["distributions_calculated"] = len(
                distribution_results.get("pck_brightness_distributions", {})
            )

            # Step 6: Create visualizations
            if create_visualizations:
                print("\n📊 Step 6: Creating visualizations...")

                # Create comprehensive dashboard
                dashboard_path = self.visualization_creator.create_comprehensive_gt_analysis_dashboard(
                    distribution_results, True, f"{output_prefix}_dashboard.svg"
                )
                if dashboard_path:
                    results["files_created"].append(dashboard_path)

                # Create individual plots if requested
                if target_pck_scores:
                    # PCK-brightness line plot
                    line_plot_path = (
                        self.visualization_creator.create_pck_brightness_line_plot(
                            distribution_results["pck_brightness_distributions"],
                            pck_threshold,
                            target_pck_scores,
                            True,
                            f"{output_prefix}_pck_brightness_line.svg",
                        )
                    )
                    if line_plot_path:
                        results["files_created"].append(line_plot_path)

                # Joint comparison plot
                joint_plot_path = (
                    self.visualization_creator.create_joint_brightness_comparison_plot(
                        distribution_results["joint_brightness_distributions"],
                        joint_names,
                        True,
                        f"{output_prefix}_joint_comparison.svg",
                    )
                )
                if joint_plot_path:
                    results["files_created"].append(joint_plot_path)

                # Correlation heatmap
                corr_plot_path = self.visualization_creator.create_correlation_heatmap(
                    correlation_results, True, f"{output_prefix}_correlations.svg"
                )
                if corr_plot_path:
                    results["files_created"].append(corr_plot_path)

            # Step 7: Export data and results
            if export_data:
                print("\n💾 Step 7: Exporting data and results...")

                # Export correlation analysis
                corr_export_path = self.pck_analyzer.export_analysis_results(
                    correlation_results, f"{output_prefix}_correlation_analysis"
                )
                if corr_export_path:
                    results["files_created"].append(corr_export_path)

                # Export distributions
                dist_export_path = self.distribution_calculator.export_distributions(
                    distribution_results["joint_brightness_distributions"],
                    distribution_results["pck_score_distributions"],
                    distribution_results["pck_brightness_distributions"],
                    f"{output_prefix}_distributions",
                )
                if dist_export_path:
                    results["files_created"].append(dist_export_path)

                # Export brightness data
                brightness_export_path = (
                    self.brightness_extractor.export_brightness_data(
                        brightness_data, f"{output_prefix}_brightness_data.csv", True
                    )
                )
                if brightness_export_path:
                    results["files_created"].append(brightness_export_path)

                # Export joint coordinates
                coords_export_path = self.gt_loader.export_joint_coordinates_to_csv(
                    joint_names,
                    f"{output_prefix}_joint_coordinates.csv",
                    **gt_filter_kwargs,
                )
                if coords_export_path:
                    results["files_created"].append(coords_export_path)

            # Final summary
            end_time = time.time()
            results["analysis_summary"]["execution_time_seconds"] = round(
                end_time - start_time, 2
            )
            results["analysis_summary"]["files_created_count"] = len(
                results["files_created"]
            )

            print("\n✅ Complete GT analysis finished successfully!")
            print(
                f"⏱️  Execution time: {results['analysis_summary']['execution_time_seconds']} seconds"
            )
            print(
                f"📁 Files created: {results['analysis_summary']['files_created_count']}"
            )

            # Store complete results
            results["correlation_results"] = correlation_results
            results["distribution_results"] = distribution_results
            results["brightness_data"] = brightness_data
            results["joint_coordinates"] = joint_coordinates

        except Exception as e:
            error_msg = f"GT analysis failed: {str(e)}"
            results["errors"].append(error_msg)
            print(f"❌ {error_msg}")
            import traceback

            traceback.print_exc()

        return results

    def run_quick_gt_analysis(
        self,
        joint_names: List[str] = None,
        pck_threshold: str = None,
        video_path: str = None,
        bin_size: int = 5,
        **gt_filter_kwargs,
    ) -> dict:
        """Run quick GT analysis with basic outputs."""
        print(f"\n⚡ Starting quick GT analysis for joints: {joint_names}")

        start_time = time.time()
        results = {
            "dataset": self.dataset_name,
            "joint_names": joint_names,
            "quick_summary": {},
            "visualizations_created": False,
        }

        try:
            # Quick correlation analysis
            correlation_results = self.pck_analyzer.analyze_pck_brightness_correlation(
                joint_names=joint_names,
                pck_threshold=pck_threshold,
                video_path=video_path,
                **gt_filter_kwargs,
            )

            if correlation_results.get("correlations"):
                # Generate quick summary
                correlations = correlation_results["correlations"]
                strong_correlations = []

                for joint, joint_corrs in correlations.items():
                    for threshold, corr_data in joint_corrs.items():
                        if (
                            abs(corr_data["correlation"]) > 0.3
                        ):  # Moderate correlation threshold
                            strong_correlations.append(
                                {
                                    "joint": joint,
                                    "threshold": threshold,
                                    "correlation": corr_data["correlation"],
                                    "samples": corr_data["valid_samples"],
                                }
                            )

                results["quick_summary"] = {
                    "total_correlations": sum(len(jc) for jc in correlations.values()),
                    "strong_correlations": len(strong_correlations),
                    "strongest_correlation": max(
                        strong_correlations, key=lambda x: abs(x["correlation"])
                    )
                    if strong_correlations
                    else None,
                }

            # Create one basic dashboard
            distribution_results = (
                self.distribution_calculator.run_complete_distribution_analysis(
                    joint_names=joint_names,
                    pck_threshold=pck_threshold,
                    bin_size=bin_size,
                    video_path=video_path,
                    **gt_filter_kwargs,
                )
            )

            if distribution_results:
                dashboard_path = self.visualization_creator.create_comprehensive_gt_analysis_dashboard(
                    distribution_results,
                    True,
                    f"quick_gt_analysis_{self.dataset_name}.svg",
                )
                results["visualizations_created"] = bool(dashboard_path)

            end_time = time.time()
            results["execution_time"] = round(end_time - start_time, 2)

            print(
                f"✅ Quick GT analysis completed in {results['execution_time']} seconds"
            )

        except Exception as e:
            print(f"❌ Quick GT analysis failed: {e}")
            results["error"] = str(e)

        return results

    def run_joint_comparison_analysis(
        self,
        joint_groups: List[List[str]],
        group_labels: List[str] = None,
        pck_threshold: str = None,
        video_path: str = None,
        bin_size: int = 5,
        **gt_filter_kwargs,
    ) -> dict:
        """Run comparison analysis between joint groups."""
        print(f"\n🔍 Starting joint comparison analysis for {len(joint_groups)} groups")

        start_time = time.time()
        results = {
            "dataset": self.dataset_name,
            "joint_groups": joint_groups,
            "group_labels": group_labels
            or [f"Group {i + 1}" for i in range(len(joint_groups))],
            "files_created": [],
            "group_summaries": [],
        }

        try:
            # Analyze each group
            for i, (joints, label) in enumerate(
                zip(joint_groups, results["group_labels"])
            ):
                print(f"\n   Analyzing {label}: {joints}")

                # Get brightness statistics for this group
                brightness_data = (
                    self.brightness_extractor.extract_brightness_for_joints(
                        joint_names=joints, video_path=video_path, **gt_filter_kwargs
                    )
                )

                if brightness_data:
                    brightness_stats = (
                        self.brightness_extractor.calculate_joint_brightness_statistics(
                            brightness_data
                        )
                    )

                    if not brightness_stats.empty:
                        group_summary = {
                            "label": label,
                            "joints": joints,
                            "total_frames": brightness_stats["frame_count"].sum(),
                            "mean_brightness": brightness_stats[
                                "mean_brightness"
                            ].mean(),
                            "brightness_variability": brightness_stats[
                                "std_brightness"
                            ].mean(),
                            "brightest_joint": brightness_stats.loc[
                                brightness_stats["mean_brightness"].idxmax(), "joint"
                            ],
                        }
                        results["group_summaries"].append(group_summary)

            # Create comparison visualizations
            if len(joint_groups) > 1:
                # Create separate analysis for each group
                all_joints = [joint for group in joint_groups for joint in group]

                distribution_results = (
                    self.distribution_calculator.run_complete_distribution_analysis(
                        joint_names=all_joints,
                        pck_threshold=pck_threshold,
                        bin_size=bin_size,
                        video_path=video_path,
                        **gt_filter_kwargs,
                    )
                )

                if distribution_results:
                    # Create group comparison plot
                    comparison_path = self.visualization_creator.create_joint_brightness_comparison_plot(
                        distribution_results["joint_brightness_distributions"],
                        all_joints,
                        True,
                        f"joint_group_comparison_{self.dataset_name}.svg",
                    )
                    if comparison_path:
                        results["files_created"].append(comparison_path)

            end_time = time.time()
            results["execution_time"] = round(end_time - start_time, 2)

            print(
                f"✅ Joint comparison analysis completed in {results['execution_time']} seconds"
            )

        except Exception as e:
            print(f"❌ Joint comparison analysis failed: {e}")
            results["error"] = str(e)

        return results

    def display_gt_workflow_status(self):
        """Display the status of all GT workflow components."""
        print(f"\n🎭 GT WORKFLOW STATUS - Dataset: {self.dataset_name}")
        print("=" * 70)

        components = [
            ("📋 GT Data Loader", self.gt_loader),
            ("🔆 Joint Brightness Extractor", self.brightness_extractor),
            ("📊 GT PCK Brightness Analyzer", self.pck_analyzer),
            ("📈 GT Distribution Calculator", self.distribution_calculator),
            ("🎨 GT Visualization Creator", self.visualization_creator),
        ]

        for name, component in components:
            status = "✅ Ready" if component else "❌ Not Available"
            print(f"{name}: {status}")

        print(f"\n🗂️  Dataset Configuration: {self.dataset_name}")

        try:
            from config import ConfigManager

            config = ConfigManager.load_config(self.dataset_name)
            print(f"📁 Save Folder: {config.save_folder}")

            if hasattr(config.paths, "ground_truth_file"):
                print(f"📄 Ground Truth File: {config.paths.ground_truth_file}")

            if hasattr(config, "video_folder"):
                print(f"📹 Video Folder: {config.video_folder}")

        except Exception as e:
            print(f"⚠️  Configuration issue: {e}")


def main():
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Master GT Analysis Orchestrator")
    parser.add_argument("dataset", help="Dataset name (e.g., 'humaneva', 'movi')")
    parser.add_argument(
        "joints", nargs="*", help="Joints to analyze (default: all available)"
    )
    parser.add_argument("--pck-threshold", help="Specific PCK threshold to analyze")
    parser.add_argument(
        "--pck-scores", nargs="*", type=int, help="Specific PCK scores to analyze"
    )
    parser.add_argument("--video", help="Path to specific video file")
    parser.add_argument("--subject", help="Filter ground truth by subject")
    parser.add_argument("--action", help="Filter ground truth by action")
    parser.add_argument("--camera", type=int, help="Filter ground truth by camera")
    parser.add_argument("--bin-size", type=int, default=5, help="Brightness bin size")
    parser.add_argument(
        "--radius", type=int, default=3, help="Sampling radius around joint"
    )
    parser.add_argument(
        "--analysis-type",
        choices=["complete", "quick", "joint_comparison"],
        default="complete",
        help="Type of analysis to run",
    )
    parser.add_argument(
        "--output-prefix", default="gt_analysis", help="Prefix for output files"
    )
    parser.add_argument(
        "--no-viz", action="store_true", help="Skip visualization creation"
    )
    parser.add_argument("--no-export", action="store_true", help="Skip data export")
    parser.add_argument(
        "--status", action="store_true", help="Show workflow status and exit"
    )

    args = parser.parse_args()

    try:
        orchestrator = MasterGTAnalysisOrchestrator(args.dataset)

        if args.status:
            orchestrator.display_gt_workflow_status()
            return

        # Set up ground truth filters
        gt_filters = {}
        if args.subject:
            gt_filters["subject"] = args.subject
        if args.action:
            gt_filters["action"] = args.action
        if args.camera is not None:
            gt_filters["camera"] = args.camera

        if args.analysis_type == "complete":
            results = orchestrator.run_complete_gt_analysis(
                joint_names=args.joints or None,
                pck_threshold=args.pck_threshold,
                video_path=args.video,
                target_pck_scores=args.pck_scores,
                bin_size=args.bin_size,
                sampling_radius=args.radius,
                create_visualizations=not args.no_viz,
                export_data=not args.no_export,
                output_prefix=args.output_prefix,
                **gt_filters,
            )

            print("\n📊 FINAL RESULTS:")
            print(f"   Files created: {len(results['files_created'])}")
            print(f"   Analysis components: {len(results['analysis_summary'])}")
            if results["errors"]:
                print(f"   Errors: {len(results['errors'])}")
                for error in results["errors"]:
                    print(f"     - {error}")

        elif args.analysis_type == "quick":
            results = orchestrator.run_quick_gt_analysis(
                joint_names=args.joints or None,
                pck_threshold=args.pck_threshold,
                video_path=args.video,
                bin_size=args.bin_size,
                **gt_filters,
            )

            print("\n📊 QUICK RESULTS:")
            if "quick_summary" in results:
                for key, value in results["quick_summary"].items():
                    print(f"   {key}: {value}")

        elif args.analysis_type == "joint_comparison":
            # For comparison, split joints into groups (example: every 2 joints)
            joints = args.joints or ["LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE"]
            joint_groups = [joints[i : i + 2] for i in range(0, len(joints), 2)]

            results = orchestrator.run_joint_comparison_analysis(
                joint_groups=joint_groups,
                pck_threshold=args.pck_threshold,
                video_path=args.video,
                bin_size=args.bin_size,
                **gt_filters,
            )

            print("\n📊 COMPARISON RESULTS:")
            print(f"   Groups analyzed: {len(results['group_summaries'])}")
            for summary in results["group_summaries"]:
                print(
                    f"   {summary['label']}: {summary['total_frames']} frames, "
                    f"avg brightness: {summary['mean_brightness']:.2f}, "
                    f"brightest: {summary['brightest_joint']}"
                )

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

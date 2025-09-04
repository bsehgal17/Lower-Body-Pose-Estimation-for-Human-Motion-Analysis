#!/usr/bin/env python3
"""
Joint Analysis CLI

Command-line interface for running joint-wise pose estimation analysis.
Creates scatter and line plots similar to per-frame analysis, focusing on individual joints.
Uses PCK values from Jointwise Metrics sheet and ground truth coordinates for brightness analysis.

Usage:
    python joint_analysis_cli.py --dataset <dataset_name> --config <config_file>
    python joint_analysis_cli.py --dataset movi --joints LEFT_HIP RIGHT_HIP LEFT_KNEE
    python joint_analysis_cli.py --dataset humaneva --all-joints --thresholds 0.01 0.05 0.1
"""

import argparse
import sys
import os
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

# Add Analysis scripts to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import ConfigManager
from analyzers.joint_brightness_analyzer import JointBrightnessAnalyzer
from visualizers.joint_brightness_visualizer import JointBrightnessVisualizer
from core.data_processor import DataProcessor
from utils import ProgressTracker, PerformanceMonitor
from datetime import datetime


class JointAnalysisCLI:
    """Command-line interface for joint analysis."""

    def __init__(self):
        """Initialize the CLI."""
        self.dataset_name = None
        self.config = None
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = None

    def parse_arguments(self) -> argparse.Namespace:
        """Parse command-line arguments."""
        parser = argparse.ArgumentParser(
            description="Joint Analysis CLI for Pose Estimation",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Basic joint analysis for MoVi dataset
  python joint_analysis_cli.py --dataset movi

  # Analyze specific joints
  python joint_analysis_cli.py --dataset humaneva --joints LEFT_HIP RIGHT_HIP LEFT_KNEE RIGHT_KNEE

  # Analyze all available joints
  python joint_analysis_cli.py --dataset movi --all-joints

  # Specify PCK thresholds
  python joint_analysis_cli.py --dataset humaneva --thresholds 0.01 0.05 0.1 --sampling-radius 5

  # Custom output directory
  python joint_analysis_cli.py --dataset movi --output results/joint_analysis

  # Generate specific plot types
  python joint_analysis_cli.py --dataset humaneva --plots scatter line heatmap

  # Ground truth analysis with custom file
  python joint_analysis_cli.py --dataset movi --ground-truth /path/to/gt.json
            """,
        )

        # Required arguments
        parser.add_argument(
            "--dataset",
            type=str,
            required=True,
            choices=["movi", "humaneva"],
            help="Dataset name (movi or humaneva)",
        )

        # Joint selection
        joint_group = parser.add_mutually_exclusive_group()
        joint_group.add_argument(
            "--joints",
            nargs="+",
            type=str,
            help="Specific joint names to analyze (e.g., LEFT_HIP RIGHT_HIP LEFT_KNEE)",
        )
        joint_group.add_argument(
            "--all-joints",
            action="store_true",
            help="Analyze all available joints in the dataset",
        )

        # PCK thresholds
        parser.add_argument(
            "--thresholds",
            nargs="+",
            type=float,
            default=[0.01, 0.05, 0.1],
            help="PCK thresholds to analyze (default: 0.01 0.05 0.1)",
        )

        # Ground truth configuration
        parser.add_argument(
            "--ground-truth",
            type=str,
            help="Path to ground truth coordinates file (overrides config)",
        )

        # Analysis parameters
        parser.add_argument(
            "--sampling-radius",
            type=int,
            default=3,
            help="Radius for brightness sampling around joint coordinates (default: 3)",
        )

        # Plot types
        parser.add_argument(
            "--plots",
            nargs="+",
            type=str,
            default=["scatter", "line", "heatmap", "summary"],
            choices=[
                "scatter",
                "line",
                "heatmap",
                "summary",
                "comparison",
                "correlation",
            ],
            help="Types of plots to generate (default: all)",
        )

        # Output configuration
        parser.add_argument(
            "--output",
            type=str,
            help="Custom output directory (default: dataset save folder)",
        )

        parser.add_argument(
            "--no-save",
            action="store_true",
            help="Don't save plots, only display them",
        )

        # Analysis options
        parser.add_argument(
            "--correlation-window",
            type=int,
            default=50,
            help="Window size for rolling correlation analysis (default: 50)",
        )

        parser.add_argument(
            "--verbose",
            action="store_true",
            help="Enable verbose output",
        )

        parser.add_argument(
            "--debug",
            action="store_true",
            help="Enable debug mode with detailed error messages",
        )

        return parser.parse_args()

    def setup_configuration(self, args: argparse.Namespace) -> bool:
        """Set up dataset configuration."""
        try:
            self.dataset_name = args.dataset
            print(
                f"🔧 Loading configuration for {self.dataset_name.upper()} dataset..."
            )

            # Load dataset configuration
            self.config = ConfigManager.load_config(self.dataset_name)

            # Add ground truth file to config if provided
            if args.ground_truth:
                if os.path.exists(args.ground_truth):
                    self.config.ground_truth_file = args.ground_truth
                    print(f"   Using custom ground truth file: {args.ground_truth}")
                else:
                    print(f"❌ Ground truth file not found: {args.ground_truth}")
                    return False

            # Set up output directory
            if args.output:
                self.output_dir = args.output
            else:
                self.output_dir = os.path.join(
                    self.config.save_folder, f"joint_analysis_{self.timestamp}"
                )

            os.makedirs(self.output_dir, exist_ok=True)
            print(f"   Output directory: {self.output_dir}")

            # Validate configuration
            if not self.config.validate():
                print("❌ Configuration validation failed")
                return False

            print("✅ Configuration loaded successfully")
            return True

        except Exception as e:
            print(f"❌ Error setting up configuration: {e}")
            if args.debug:
                import traceback

                traceback.print_exc()
            return False

    def determine_joints_to_analyze(self, args: argparse.Namespace) -> List[str]:
        """Determine which joints to analyze based on arguments."""
        if args.joints:
            return args.joints

        # Default joints for different datasets
        default_joints = {
            "movi": [
                "LEFT_HIP",
                "RIGHT_HIP",
                "LEFT_KNEE",
                "RIGHT_KNEE",
                "LEFT_ANKLE",
                "RIGHT_ANKLE",
            ],
            "humaneva": [
                "LEFT_HIP",
                "RIGHT_HIP",
                "LEFT_KNEE",
                "RIGHT_KNEE",
                "LEFT_ANKLE",
                "RIGHT_ANKLE",
            ],
        }

        if args.all_joints:
            # Load PCK data to discover all available joints
            try:
                data_processor = DataProcessor(self.config)
                pck_df = data_processor.load_pck_per_frame_scores()
                if pck_df is not None:
                    # Extract joint names from jointwise PCK columns
                    jointwise_columns = [
                        col for col in pck_df.columns if "jointwise_pck" in col.lower()
                    ]

                    joints = set()
                    for col in jointwise_columns:
                        # Parse joint name from column (e.g., "LEFT_HIP_jointwise_pck_0.01")
                        parts = col.split("_")
                        joint_parts = []
                        for part in parts:
                            if part.lower() == "jointwise":
                                break
                            joint_parts.append(part)

                        if joint_parts:
                            joint_name = "_".join(joint_parts)
                            joints.add(joint_name)

                    if joints:
                        return sorted(list(joints))
            except Exception as e:
                print(f"⚠️  Could not auto-detect joints: {e}")

        # Fall back to default joints
        return default_joints.get(self.dataset_name, default_joints["movi"])

    def load_and_validate_data(self) -> Optional[pd.DataFrame]:
        """Load and validate PCK data for analysis."""
        try:
            print("📊 Loading PCK data...")
            data_processor = DataProcessor(self.config)

            # Load per-frame PCK data (contains jointwise metrics)
            pck_df = data_processor.load_pck_per_frame_scores()

            if pck_df is None:
                print("❌ Could not load PCK data")
                return None

            # Check for jointwise PCK columns
            jointwise_columns = [
                col for col in pck_df.columns if "jointwise_pck" in col.lower()
            ]

            if not jointwise_columns:
                print("❌ No jointwise PCK columns found in data")
                print(f"   Available columns: {list(pck_df.columns)}")
                return None

            print(f"✅ Loaded PCK data with {len(pck_df)} frames")
            print(f"   Found {len(jointwise_columns)} jointwise PCK columns")

            return pck_df

        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None

    @PerformanceMonitor.timing_decorator
    def run_joint_analysis(
        self,
        joints: List[str],
        thresholds: List[float],
        sampling_radius: int,
        pck_data: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Run joint brightness analysis."""
        print(f"🔍 Running joint analysis for {len(joints)} joints...")
        print(f"   Joints: {', '.join(joints)}")
        print(f"   PCK thresholds: {thresholds}")
        print(f"   Sampling radius: {sampling_radius}")

        try:
            # Create joint brightness analyzer
            analyzer = JointBrightnessAnalyzer(
                config=self.config, joint_names=joints, sampling_radius=sampling_radius
            )

            # Run analysis
            with ProgressTracker("Joint analysis") as tracker:
                tracker.update("Analyzing joint brightness patterns...")
                analysis_results = analyzer.analyze(pck_data)

            if not analysis_results:
                print("❌ Analysis returned no results")
                return {}

            # Filter results by requested thresholds
            filtered_results = {}
            for metric_name, results in analysis_results.items():
                threshold_val = float(results.get("threshold", "0"))

                if any(abs(threshold_val - t) < 0.001 for t in thresholds):
                    filtered_results[metric_name] = results

            print(f"✅ Analysis completed with {len(filtered_results)} metrics")
            return filtered_results

        except Exception as e:
            print(f"❌ Error during analysis: {e}")
            import traceback

            traceback.print_exc()
            return {}

    def create_visualizations(
        self,
        analysis_results: Dict[str, Any],
        plot_types: List[str],
        correlation_window: int,
        save_plots: bool,
    ):
        """Create visualizations for analysis results."""
        if not analysis_results:
            print("❌ No analysis results to visualize")
            return

        print("📈 Creating visualizations...")
        print(f"   Plot types: {', '.join(plot_types)}")

        try:
            # Create visualizer
            visualizer = JointBrightnessVisualizer(self.config)

            # Base save path (None if not saving)
            save_path = (
                os.path.join(self.output_dir, "joint_analysis") if save_plots else None
            )

            # Create different types of plots
            if "scatter" in plot_types:
                print("   Creating scatter plots...")
                visualizer.create_per_frame_scatter_plots(analysis_results, save_path)

            if "line" in plot_types:
                print("   Creating line plots...")
                visualizer.create_per_frame_line_plots(analysis_results, save_path)

            if "heatmap" in plot_types:
                print("   Creating heatmap...")
                visualizer.create_brightness_heatmap(analysis_results, save_path)

            if "summary" in plot_types:
                print("   Creating summary plots...")
                visualizer.create_combined_summary_plot(analysis_results, save_path)

            if "comparison" in plot_types:
                print("   Creating comparison plots...")
                visualizer.create_joint_comparison_plot(analysis_results, save_path)

            if "correlation" in plot_types:
                print("   Creating correlation over time plots...")
                visualizer.create_joint_correlation_over_time(
                    analysis_results, save_path, correlation_window
                )

            print("✅ Visualizations completed")

        except Exception as e:
            print(f"❌ Error creating visualizations: {e}")
            import traceback

            traceback.print_exc()

    def generate_analysis_report(
        self, analysis_results: Dict[str, Any], args: argparse.Namespace
    ):
        """Generate a comprehensive analysis report."""
        if not analysis_results:
            return

        print("\n" + "=" * 70)
        print("📋 JOINT ANALYSIS REPORT")
        print("=" * 70)

        print(f"Dataset: {self.dataset_name.upper()}")
        print(f"Timestamp: {self.timestamp}")
        print(f"Output Directory: {self.output_dir}")
        print(f"Number of Metrics Analyzed: {len(analysis_results)}")

        # Organize results by joint
        joint_summaries = {}
        for metric_name, results in analysis_results.items():
            joint_name = results.get("joint_name", "Unknown")
            threshold = results.get("threshold", "Unknown")

            if joint_name not in joint_summaries:
                joint_summaries[joint_name] = {}

            joint_summaries[joint_name][threshold] = results

        print(f"\nAnalyzed Joints: {', '.join(joint_summaries.keys())}")
        print(
            f"PCK Thresholds: {set(r.get('threshold', 'Unknown') for r in analysis_results.values())}"
        )

        # Joint-wise summary
        print("\n" + "-" * 50)
        print("JOINT-WISE SUMMARY")
        print("-" * 50)

        for joint_name, thresholds in joint_summaries.items():
            print(f"\n🔸 {joint_name}:")

            for threshold, results in thresholds.items():
                correlation = results.get("correlation", {}).get("pearson", 0.0)
                total_frames = results.get("total_frames", 0)
                brightness_stats = results.get("brightness_stats", {}).get(
                    "overall", {}
                )
                mean_brightness = brightness_stats.get("mean", 0.0)

                print(f"  Threshold {threshold}:")
                print(f"    • Frames analyzed: {total_frames}")
                print(f"    • PCK-Brightness correlation: {correlation:.3f}")
                print(f"    • Mean brightness: {mean_brightness:.1f}")

                # Score range analysis
                score_ranges = results.get("score_ranges", {})
                if score_ranges:
                    print("    • Score range analysis:")
                    for range_name, range_data in score_ranges.items():
                        count = range_data.get("count", 0)
                        mean_br = range_data.get("mean_brightness", 0.0)
                        print(
                            f"      - {range_name}: {count} frames, avg brightness {mean_br:.1f}"
                        )

        # Overall statistics
        print("\n" + "-" * 50)
        print("OVERALL STATISTICS")
        print("-" * 50)

        all_correlations = [
            r.get("correlation", {}).get("pearson", 0.0)
            for r in analysis_results.values()
        ]
        all_frame_counts = [r.get("total_frames", 0) for r in analysis_results.values()]

        if all_correlations:
            print(
                f"Average correlation across all metrics: {np.mean(all_correlations):.3f}"
            )
            print(
                f"Correlation range: {np.min(all_correlations):.3f} to {np.max(all_correlations):.3f}"
            )

        if all_frame_counts:
            print(f"Total frames analyzed: {sum(all_frame_counts)}")
            print(f"Average frames per metric: {np.mean(all_frame_counts):.0f}")

        # Save report to file
        if not args.no_save:
            report_file = os.path.join(self.output_dir, "analysis_report.txt")
            try:
                with open(report_file, "w") as f:
                    # Write the same report to file
                    f.write("JOINT ANALYSIS REPORT\n")
                    f.write("=" * 70 + "\n")
                    f.write(f"Dataset: {self.dataset_name.upper()}\n")
                    f.write(f"Timestamp: {self.timestamp}\n")
                    f.write(f"Number of Metrics Analyzed: {len(analysis_results)}\n\n")

                    for joint_name, thresholds in joint_summaries.items():
                        f.write(f"\n{joint_name}:\n")
                        for threshold, results in thresholds.items():
                            correlation = results.get("correlation", {}).get(
                                "pearson", 0.0
                            )
                            total_frames = results.get("total_frames", 0)
                            brightness_stats = results.get("brightness_stats", {}).get(
                                "overall", {}
                            )
                            mean_brightness = brightness_stats.get("mean", 0.0)

                            f.write(f"  Threshold {threshold}:\n")
                            f.write(f"    Frames: {total_frames}\n")
                            f.write(f"    Correlation: {correlation:.3f}\n")
                            f.write(f"    Mean brightness: {mean_brightness:.1f}\n")

                print(f"\n📄 Report saved to: {report_file}")
            except Exception as e:
                print(f"⚠️  Could not save report: {e}")

        print("\n" + "=" * 70)

    def main(self):
        """Main CLI execution."""
        print("🚀 Joint Analysis CLI for Pose Estimation")
        print("=" * 50)

        # Parse arguments
        args = self.parse_arguments()

        # Setup configuration
        if not self.setup_configuration(args):
            sys.exit(1)

        # Determine joints to analyze
        joints_to_analyze = self.determine_joints_to_analyze(args)
        print(f"📍 Joints to analyze: {', '.join(joints_to_analyze)}")

        # Load and validate data
        pck_data = self.load_and_validate_data()
        if pck_data is None:
            sys.exit(1)

        # Run analysis
        analysis_results = self.run_joint_analysis(
            joints=joints_to_analyze,
            thresholds=args.thresholds,
            sampling_radius=args.sampling_radius,
            pck_data=pck_data,
        )

        if not analysis_results:
            print("❌ No analysis results generated")
            sys.exit(1)

        # Create visualizations
        self.create_visualizations(
            analysis_results=analysis_results,
            plot_types=args.plots,
            correlation_window=args.correlation_window,
            save_plots=not args.no_save,
        )

        # Generate report
        self.generate_analysis_report(analysis_results, args)

        print("\n✅ Joint analysis completed successfully!")
        if not args.no_save:
            print(f"📁 Results saved to: {self.output_dir}")


if __name__ == "__main__":
    cli = JointAnalysisCLI()
    try:
        cli.main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

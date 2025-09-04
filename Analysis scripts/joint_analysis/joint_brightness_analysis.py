"""
Joint Brightness Analysis Pipeline.

This module provides the JointBrightnessAnalysisPipeline class for analyzing
the relationship between jointwise PCK scores and brightness values from per-frame data.
"""

from datetime import datetime
from utils.performance_utils import PerformanceMonitor
from core.data_processor import DataProcessor
from visualizers.visualization_factory import VisualizationFactory
from analyzers.analyzer_factory import AnalyzerFactory
from config import ConfigManager
import sys
import os

# Add the Analysis scripts directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class JointBrightnessAnalysisPipeline:
    """Pipeline for joint brightness analysis."""

    def __init__(
        self, dataset_name: str, joint_names: list = None, sampling_radius: int = 3
    ):
        """Initialize the analysis pipeline."""
        self.dataset_name = dataset_name
        self.joint_names = joint_names or self._get_default_joints()
        self.sampling_radius = sampling_radius
        self.config = ConfigManager.load_config(dataset_name)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create save folder
        os.makedirs(self.config.save_folder, exist_ok=True)

        # Initialize components
        self.data_processor = DataProcessor(self.config)

        # Create analyzer with joint names
        self.analyzer = AnalyzerFactory.create_analyzer(
            "joint_brightness",
            self.config,
            joint_names=self.joint_names,
            sampling_radius=self.sampling_radius,
        )

        self.visualizer = VisualizationFactory.create_visualizer(
            "joint_brightness", self.config
        )

    def _get_default_joints(self):
        """Get default joint names for analysis."""
        # Lower body joints as specified in the request
        return [
            "LEFT_HIP",
            "RIGHT_HIP",
            "LEFT_KNEE",
            "RIGHT_KNEE",
            "LEFT_ANKLE",
            "RIGHT_ANKLE",
        ]

    @PerformanceMonitor.timing_decorator
    def run_analysis(self, save_plots: bool = True, per_frame_only: bool = False):
        """Run the complete joint brightness analysis."""
        print(
            f"Starting Joint Brightness Analysis for {self.dataset_name.upper()} dataset..."
        )
        print(f"Joints: {', '.join(self.joint_names)}")
        print(f"Timestamp: {self.timestamp}")
        print(f"Save folder: {self.config.save_folder}")
        print(f"Sampling radius: {self.sampling_radius}")
        if per_frame_only:
            print("Mode: Per-frame plots only")
        print("=" * 80)

        # Step 1: Load per-frame data with jointwise PCK scores
        print("Step 1: Loading per-frame data with jointwise PCK scores...")
        per_frame_data = self._load_jointwise_pck_data()

        if per_frame_data is None or per_frame_data.empty:
            print("❌ Error: No per-frame data with jointwise PCK scores found.")
            return None

        print(f"✅ Loaded {len(per_frame_data)} per-frame records")

        # Find jointwise PCK columns
        jointwise_columns = [
            col for col in per_frame_data.columns if "jointwise_pck" in col.lower()
        ]
        print(f"   Found {len(jointwise_columns)} jointwise PCK columns")
        for col in jointwise_columns[:5]:  # Show first 5
            print(f"     - {col}")
        if len(jointwise_columns) > 5:
            print(f"     ... and {len(jointwise_columns) - 5} more")
        print()

        # Step 2: Run joint brightness analysis
        print("Step 2: Running joint brightness analysis...")
        analysis_results = self.analyzer.analyze(per_frame_data)

        if not analysis_results:
            print("❌ Error: No analysis results generated.")
            return None

        print(
            f"✅ Analysis completed for {len(analysis_results)} joint-threshold combinations"
        )

        # Print summary statistics
        self._print_analysis_summary(analysis_results)
        print()

        # Step 3: Create visualizations
        if save_plots:
            print("Step 3: Creating visualizations...")
            save_path = os.path.join(
                self.config.save_folder,
                f"joint_brightness_{self.dataset_name}_{self.timestamp}",
            )

            if per_frame_only:
                # Create only per-frame plots
                print("Creating per-frame visualizations only...")
                self.visualizer.create_per_frame_scatter_plots(
                    analysis_results, save_path
                )
                self.visualizer.create_per_frame_line_plots(analysis_results, save_path)
                self.visualizer.create_joint_correlation_over_time(
                    analysis_results, save_path
                )
            else:
                # Create all plots
                # Create individual plots for each joint-threshold combination
                self.visualizer.create_plot(analysis_results, save_path)

                # Create combined summary plot
                self.visualizer.create_combined_summary_plot(
                    analysis_results, save_path
                )

                # Create joint comparison plots
                self.visualizer.create_joint_comparison_plot(
                    analysis_results, save_path
                )

                # Create brightness heatmap
                self.visualizer.create_brightness_heatmap(analysis_results, save_path)

                # Create per-frame scatter plots (similar to PCK analysis)
                self.visualizer.create_per_frame_scatter_plots(
                    analysis_results, save_path
                )

                # Create per-frame line plots showing evolution over time
                self.visualizer.create_per_frame_line_plots(analysis_results, save_path)

                # Create correlation over time plots
                self.visualizer.create_joint_correlation_over_time(
                    analysis_results, save_path
                )

            print("✅ All visualizations created successfully")
        else:
            print("Step 3: Skipping visualizations (save_plots=False)")

        # Step 4: Export results
        print("Step 4: Exporting results...")
        self.export_results_to_csv(analysis_results)

        print("\n" + "=" * 80)
        print(
            f"Joint Brightness Analysis completed for {self.dataset_name.upper()} dataset!"
        )

        return analysis_results

    def _load_jointwise_pck_data(self):
        """Load per-frame data containing jointwise PCK scores."""
        try:
            # Check for a specific sheet name "Jointwise Metrics" as mentioned in the request
            pck_file_path = getattr(self.config.paths, "pck_file_path", None)

            if not pck_file_path or not os.path.exists(pck_file_path):
                print(f"❌ PCK file not found: {pck_file_path}")
                return None

            # Try to load from "Jointwise Metrics" sheet first
            try:
                import pandas as pd

                data = pd.read_excel(pck_file_path, sheet_name="Jointwise Metrics")
                print("✅ Loaded data from 'Jointwise Metrics' sheet")
                return data
            except Exception as e:
                print(f"   Could not load 'Jointwise Metrics' sheet: {e}")

            # Fallback to regular per-frame data loading
            return self.data_processor.load_pck_per_frame_scores()

        except Exception as e:
            print(f"❌ Error loading jointwise PCK data: {e}")
            return None

    def _print_analysis_summary(self, analysis_results):
        """Print summary of analysis results."""
        print("\n📊 Joint Brightness Analysis Summary:")
        print("-" * 50)

        # Group by joint
        joint_summaries = {}
        for metric_name, results in analysis_results.items():
            if not results or "joint_name" not in results:
                continue

            joint_name = results["joint_name"]
            threshold = results["threshold"]

            if joint_name not in joint_summaries:
                joint_summaries[joint_name] = {}

            joint_summaries[joint_name][threshold] = results

        for joint_name, thresholds in joint_summaries.items():
            print(f"\n{joint_name}:")

            for threshold, results in thresholds.items():
                total_frames = results["total_frames"]
                correlation = results.get("correlation", {}).get("pearson", 0.0)
                mean_brightness = (
                    results.get("brightness_stats", {})
                    .get("overall", {})
                    .get("mean", 0.0)
                )

                print(f"  • Threshold {threshold}:")
                print(f"    - Frames analyzed: {total_frames}")
                print(f"    - PCK-Brightness correlation: {correlation:.3f}")
                print(f"    - Mean brightness: {mean_brightness:.1f}")

                # Score range info
                score_ranges = results.get("score_ranges", {})
                if score_ranges:
                    for range_name, range_data in score_ranges.items():
                        count = range_data["count"]
                        mean_bright = range_data["mean_brightness"]
                        print(
                            f"    - {range_name.capitalize()} PCK scores: {count} frames, brightness: {mean_bright:.1f}"
                        )

    def export_results_to_csv(self, analysis_results, filename: str = None):
        """Export analysis results to CSV for further analysis."""
        import pandas as pd

        if filename is None:
            filename = (
                f"joint_brightness_results_{self.dataset_name}_{self.timestamp}.csv"
            )

        export_data = []

        for metric_name, results in analysis_results.items():
            if not results or "joint_name" not in results:
                continue

            joint_name = results["joint_name"]
            threshold = results["threshold"]
            correlation = results.get("correlation", {}).get("pearson", 0.0)
            total_frames = results["total_frames"]

            # Overall brightness stats
            brightness_stats = results.get("brightness_stats", {}).get("overall", {})

            export_row = {
                "metric_name": metric_name,
                "joint_name": joint_name,
                "pck_threshold": threshold,
                "total_frames": total_frames,
                "pck_brightness_correlation": correlation,
                "mean_brightness": brightness_stats.get("mean", 0.0),
                "std_brightness": brightness_stats.get("std", 0.0),
                "min_brightness": brightness_stats.get("min", 0.0),
                "max_brightness": brightness_stats.get("max", 0.0),
                "median_brightness": brightness_stats.get("median", 0.0),
            }

            # Add score range data
            score_ranges = results.get("score_ranges", {})
            for range_name, range_data in score_ranges.items():
                export_row[f"{range_name}_pck_frame_count"] = range_data["count"]
                export_row[f"{range_name}_pck_mean_brightness"] = range_data[
                    "mean_brightness"
                ]
                export_row[f"{range_name}_pck_std_brightness"] = range_data[
                    "std_brightness"
                ]

            export_data.append(export_row)

        if export_data:
            df = pd.DataFrame(export_data)
            csv_path = os.path.join(self.config.save_folder, filename)
            df.to_csv(csv_path, index=False)
            print(f"✅ Results exported to: {csv_path}")
            return csv_path
        else:
            print("❌ No data to export")
            return None

    def generate_summary_report(self, analysis_results):
        """Generate a summary report of the analysis."""
        if not analysis_results:
            return ""

        report_lines = []
        report_lines.append("Joint Brightness Analysis Report")
        report_lines.append(f"Dataset: {self.dataset_name.upper()}")
        report_lines.append(f"Timestamp: {self.timestamp}")
        report_lines.append(f"Joints Analyzed: {', '.join(self.joint_names)}")
        report_lines.append(f"Sampling Radius: {self.sampling_radius} pixels")
        report_lines.append("=" * 60)

        # Summary statistics
        all_correlations = []
        all_brightness = []
        total_frames = 0

        for metric_name, results in analysis_results.items():
            if results and "correlation" in results:
                all_correlations.append(results["correlation"]["pearson"])
                all_brightness.append(results["brightness_stats"]["overall"]["mean"])
                total_frames += results["total_frames"]

        if all_correlations:
            import numpy as np

            report_lines.append("\nOverall Summary:")
            report_lines.append(f"  Total frames analyzed: {total_frames}")
            report_lines.append(
                f"  Average correlation: {np.mean(all_correlations):.3f}"
            )
            report_lines.append(
                f"  Correlation range: {np.min(all_correlations):.3f} to {np.max(all_correlations):.3f}"
            )
            report_lines.append(f"  Average brightness: {np.mean(all_brightness):.1f}")
            report_lines.append(
                f"  Brightness range: {np.min(all_brightness):.1f} to {np.max(all_brightness):.1f}"
            )

        # Detailed results
        report_lines.append("\nDetailed Results:")
        for metric_name, results in analysis_results.items():
            if not results:
                continue

            joint_name = results.get("joint_name", "Unknown")
            threshold = results.get("threshold", "0.05")
            correlation = results.get("correlation", {}).get("pearson", 0.0)
            frames = results.get("total_frames", 0)

            report_lines.append(f"\n{joint_name} (PCK @ {threshold}):")
            report_lines.append(f"  Correlation: {correlation:.3f}")
            report_lines.append(f"  Frames: {frames}")

            # Score ranges
            score_ranges = results.get("score_ranges", {})
            for range_name, range_data in score_ranges.items():
                count = range_data["count"]
                mean_bright = range_data["mean_brightness"]
                report_lines.append(
                    f"  {range_name.capitalize()}: {count} frames, brightness {mean_bright:.1f}"
                )

        report_text = "\n".join(report_lines)

        # Save report
        report_filename = (
            f"joint_brightness_report_{self.dataset_name}_{self.timestamp}.txt"
        )
        report_path = os.path.join(self.config.save_folder, report_filename)

        with open(report_path, "w") as f:
            f.write(report_text)

        print(f"✅ Summary report saved to: {report_path}")
        return report_text

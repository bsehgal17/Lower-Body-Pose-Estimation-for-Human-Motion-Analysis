#!/usr/bin/env python3
"""
Joint Analysis Script

Simple script for running joint-wise pose estimation analysis.
Creates scatter and line plots focusing on individual joints.
Uses PCK values from Jointwise Metrics sheet and ground truth coordinates for brightness analysis.

Usage:
    1. Change the DATASET_NAME variable below to your desired dataset
    2. Run: python joint_analysis_script.py

No command-line arguments or interactive input needed.
"""

import sys
import os
from typing import Dict, Any
import pandas as pd
from pathlib import Path

# Add Analysis scripts to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import ConfigManager
from analyzers.joint_brightness_analyzer import JointBrightnessAnalyzer
from visualizers.joint_brightness_visualizer import JointBrightnessVisualizer
from core.data_processor import DataProcessor
from utils import ProgressTracker
from datetime import datetime

# ============================================================================
# CONFIGURATION - MODIFY THESE VALUES AS NEEDED
# ============================================================================

# Dataset to analyze - change this to "movi" or "humaneva"
DATASET_NAME = "movi"

# Joints to analyze (hip, knee, ankle - both left and right)
JOINTS_TO_ANALYZE = [
    "LEFT_HIP",
    "RIGHT_HIP",
    "LEFT_KNEE",
    "RIGHT_KNEE",
    "LEFT_ANKLE",
    "RIGHT_ANKLE",
]

# PCK thresholds for analysis
PCK_THRESHOLDS = [0.01, 0.05, 0.1]

# Sampling radius for brightness analysis at joint locations
SAMPLING_RADIUS = 3

# Plot types to generate
PLOT_TYPES = ["scatter", "line"]

# Correlation window size for time-series analysis
CORRELATION_WINDOW = 30

# Whether to save results to files
SAVE_RESULTS = True

# Output directory (None for auto-generated timestamp directory)
OUTPUT_DIR = None

# ============================================================================
# SCRIPT CLASS
# ============================================================================


class JointAnalysisScript:
    """Simple script for joint analysis without CLI complexity."""

    def __init__(self):
        """Initialize the script."""
        self.dataset_name = DATASET_NAME
        self.config = None
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = None

    def setup_configuration(self) -> bool:
        """Setup configuration for the specified dataset."""
        try:
            print(f"Setting up configuration for dataset: {self.dataset_name}")

            # Initialize config manager
            config_manager = ConfigManager()

            # Load dataset configuration
            self.config = config_manager.get_dataset_config(self.dataset_name)

            if self.config is None:
                print(
                    f"ERROR: Could not load configuration for dataset '{self.dataset_name}'"
                )
                print("Available datasets: movi, humaneva")
                return False

            # Setup output directory
            if OUTPUT_DIR:
                self.output_dir = Path(OUTPUT_DIR)
            else:
                self.output_dir = Path(
                    f"joint_analysis_{self.dataset_name}_{self.timestamp}"
                )

            if SAVE_RESULTS:
                self.output_dir.mkdir(parents=True, exist_ok=True)
                print(f"Output directory: {self.output_dir}")

            return True

        except Exception as e:
            print(f"ERROR: Configuration setup failed: {e}")
            return False

    def load_and_validate_data(self) -> pd.DataFrame:
        """Load and validate PCK data."""
        try:
            print("Loading PCK data...")

            # Create data processor
            data_processor = DataProcessor(self.config)

            # Load PCK per-frame scores
            pck_df = data_processor.load_pck_per_frame_scores()

            if pck_df is None:
                print("ERROR: Could not load PCK data")
                return None

            print(f"Loaded PCK data with {len(pck_df)} frames")

            # Validate that required joints exist in data
            available_joints = set()
            for col in pck_df.columns:
                if "jointwise_pck" in col.lower():
                    # Parse joint name from column (e.g., "LEFT_HIP_jointwise_pck_0.01")
                    parts = col.split("_")
                    joint_parts = []
                    for part in parts:
                        if part.lower() == "jointwise":
                            break
                        joint_parts.append(part)

                    if joint_parts:
                        joint_name = "_".join(joint_parts)
                        available_joints.add(joint_name)

            print(f"Available joints in data: {sorted(available_joints)}")

            # Check if requested joints are available
            missing_joints = set(JOINTS_TO_ANALYZE) - available_joints
            if missing_joints:
                print(
                    f"WARNING: Some requested joints not found in data: {missing_joints}"
                )
                print("Analysis will continue with available joints only")

            return pck_df

        except Exception as e:
            print(f"ERROR: Data loading failed: {e}")
            import traceback

            traceback.print_exc()
            return None

    def run_joint_analysis(self, pck_data: pd.DataFrame) -> Dict[str, Any]:
        """Run joint brightness analysis."""
        print(f"Running joint analysis for {len(JOINTS_TO_ANALYZE)} joints...")
        print(f"   Joints: {', '.join(JOINTS_TO_ANALYZE)}")
        print(f"   PCK thresholds: {PCK_THRESHOLDS}")
        print(f"   Sampling radius: {SAMPLING_RADIUS}")

        try:
            # Create joint brightness analyzer
            analyzer = JointBrightnessAnalyzer(
                config=self.config,
                joint_names=JOINTS_TO_ANALYZE,
                sampling_radius=SAMPLING_RADIUS,
            )

            # Run analysis
            with ProgressTracker("Joint analysis") as tracker:
                tracker.update("Analyzing joint brightness patterns...")
                analysis_results = analyzer.analyze(pck_data)

            if not analysis_results:
                print("ERROR: Analysis returned no results")
                return {}

            # Filter results by requested thresholds
            filtered_results = {}
            for metric_name, results in analysis_results.items():
                threshold_val = float(results.get("threshold", "0"))

                if any(abs(threshold_val - t) < 0.001 for t in PCK_THRESHOLDS):
                    filtered_results[metric_name] = results

            print(f"Analysis completed with {len(filtered_results)} metrics")
            return filtered_results

        except Exception as e:
            print(f"ERROR: Error during analysis: {e}")
            import traceback

            traceback.print_exc()
            return {}

    def create_visualizations(self, analysis_results: Dict[str, Any]) -> None:
        """Create visualizations."""
        if not analysis_results:
            print("ERROR: No analysis results to visualize")
            return

        print("Creating visualizations...")

        try:
            # Create joint brightness visualizer
            visualizer = JointBrightnessVisualizer(config=self.config)

            # Generate plots
            with ProgressTracker("Visualization generation") as tracker:
                tracker.update("Creating plots...")

                for plot_type in PLOT_TYPES:
                    tracker.update(f"Creating {plot_type} plots...")

                    if plot_type == "scatter":
                        visualizer.create_scatter_plots(
                            analysis_results, self.output_dir
                        )
                    elif plot_type == "line":
                        visualizer.create_line_plots(analysis_results, self.output_dir)
                    elif plot_type == "heatmap":
                        visualizer.create_heatmap_plots(
                            analysis_results, self.output_dir
                        )
                    elif plot_type == "distribution":
                        visualizer.create_distribution_plots(
                            analysis_results, self.output_dir
                        )

            print("Visualizations completed")
            if SAVE_RESULTS:
                print(f"Plots saved to: {self.output_dir}")

        except Exception as e:
            print(f"ERROR: Error creating visualizations: {e}")
            import traceback

            traceback.print_exc()

    def generate_analysis_report(self, analysis_results: Dict[str, Any]) -> None:
        """Generate analysis summary report."""
        if not SAVE_RESULTS:
            return

        try:
            report_file = self.output_dir / "analysis_report.txt"

            with open(report_file, "w") as f:
                f.write("Joint Analysis Report\n")
                f.write("=" * 50 + "\n\n")

                f.write(f"Dataset: {self.dataset_name}\n")
                f.write(
                    f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                )
                f.write(f"Joints Analyzed: {', '.join(JOINTS_TO_ANALYZE)}\n")
                f.write(f"PCK Thresholds: {PCK_THRESHOLDS}\n")
                f.write(f"Sampling Radius: {SAMPLING_RADIUS}\n")
                f.write(f"Plot Types: {', '.join(PLOT_TYPES)}\n\n")

                f.write("Analysis Results Summary:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Total Metrics: {len(analysis_results)}\n")

                # Summary statistics
                for metric_name, results in analysis_results.items():
                    f.write(f"\n{metric_name}:\n")
                    if "summary_stats" in results:
                        stats = results["summary_stats"]
                        for stat_name, stat_value in stats.items():
                            f.write(f"  {stat_name}: {stat_value:.4f}\n")

            print(f"Analysis report saved to: {report_file}")

        except Exception as e:
            print(f"WARNING: Could not save report: {e}")

    def run(self) -> bool:
        """Run the complete joint analysis."""
        print("Joint Analysis Script")
        print("=" * 50)
        print(f"Dataset: {DATASET_NAME}")
        print(f"Joints: {', '.join(JOINTS_TO_ANALYZE)}")
        print("=" * 50)

        # Setup configuration
        if not self.setup_configuration():
            return False

        # Load and validate data
        pck_data = self.load_and_validate_data()
        if pck_data is None:
            return False

        # Run analysis
        analysis_results = self.run_joint_analysis(pck_data)
        if not analysis_results:
            print("ERROR: No analysis results generated")
            return False

        # Create visualizations
        if SAVE_RESULTS:
            self.create_visualizations(analysis_results)

        # Generate report
        if SAVE_RESULTS:
            self.generate_analysis_report(analysis_results)

        print("\nJoint analysis completed successfully!")
        if SAVE_RESULTS:
            print(f"Results saved to: {self.output_dir}")

        return True


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Create and run the analysis script
    script = JointAnalysisScript()

    try:
        success = script.run()
        if not success:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

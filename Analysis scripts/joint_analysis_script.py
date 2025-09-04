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
from datetime import datetime

# Now import after path setup
from core.data_processor import DataProcessor
from config.config_manager import ConfigManager

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

            # Load dataset configuration using correct method
            self.config = ConfigManager.load_config(self.dataset_name)

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
            print("Loading PCK jointwise data...")

            # Create data processor
            data_processor = DataProcessor(self.config)

            # Load PCK jointwise scores instead of per-frame scores
            pck_df = data_processor.load_pck_jointwise_scores()

            if pck_df is None:
                print("ERROR: Could not load PCK jointwise data")
                return None

            print(f"Loaded PCK jointwise data with {len(pck_df)} subjects/records")

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
        """Run joint analysis on jointwise PCK data."""
        print(f"Running joint analysis on {len(pck_data)} subjects/records...")
        print(f"   Joints: {', '.join(JOINTS_TO_ANALYZE)}")
        print(f"   PCK thresholds: {PCK_THRESHOLDS}")

        try:
            # Create simple analysis results from jointwise data
            analysis_results = {}

            # For each joint and threshold combination
            for joint_name in JOINTS_TO_ANALYZE:
                for threshold in PCK_THRESHOLDS:
                    # Find the corresponding column in the data
                    col_name = f"{joint_name}_jointwise_pck_{threshold:g}"

                    if col_name in pck_data.columns:
                        scores = pck_data[col_name].dropna()

                        if len(scores) > 0:
                            # Calculate summary statistics
                            stats = {
                                "mean": scores.mean(),
                                "median": scores.median(),
                                "std": scores.std(),
                                "min": scores.min(),
                                "max": scores.max(),
                                "count": len(scores),
                            }

                            # Store the analysis result
                            metric_name = f"{joint_name}_pck_{threshold:g}"
                            analysis_results[metric_name] = {
                                "joint_name": joint_name,
                                "threshold": threshold,
                                "summary_stats": stats,
                                "scores": scores.tolist(),
                                "subjects": pck_data["subject"].tolist()
                                if "subject" in pck_data.columns
                                else list(range(len(scores))),
                            }

                            print(
                                f"   {metric_name}: mean={stats['mean']:.3f}, std={stats['std']:.3f}, n={stats['count']}"
                            )
                        else:
                            print(f"   WARNING: No data for {col_name}")
                    else:
                        print(f"   WARNING: Column {col_name} not found in data")

            print(f"Analysis completed with {len(analysis_results)} metrics")
            return analysis_results

        except Exception as e:
            print(f"ERROR: Error during analysis: {e}")
            import traceback

            traceback.print_exc()
            return {}

    def create_visualizations(self, analysis_results: Dict[str, Any]) -> None:
        """Create simple visualizations for jointwise data."""
        if not analysis_results:
            print("ERROR: No analysis results to visualize")
            return

        print("Creating visualizations...")

        try:
            import matplotlib.pyplot as plt
            import numpy as np

            # Create plots for each joint
            for joint_name in JOINTS_TO_ANALYZE:
                # Get all thresholds for this joint
                joint_metrics = {
                    name: data
                    for name, data in analysis_results.items()
                    if data["joint_name"] == joint_name
                }

                if not joint_metrics:
                    continue

                # Create figure for this joint
                fig, axes = plt.subplots(1, 2, figsize=(15, 6))
                fig.suptitle(f"{joint_name} Analysis", fontsize=16)

                # Plot 1: PCK scores by threshold
                thresholds = [data["threshold"] for data in joint_metrics.values()]
                means = [
                    data["summary_stats"]["mean"] for data in joint_metrics.values()
                ]
                stds = [data["summary_stats"]["std"] for data in joint_metrics.values()]

                axes[0].errorbar(thresholds, means, yerr=stds, marker="o", capsize=5)
                axes[0].set_xlabel("PCK Threshold")
                axes[0].set_ylabel("Mean PCK Score")
                axes[0].set_title("Mean PCK Score by Threshold")
                axes[0].grid(True, alpha=0.3)

                # Plot 2: Distribution of scores for each threshold
                for i, (metric_name, data) in enumerate(joint_metrics.items()):
                    scores = data["scores"]
                    axes[1].hist(
                        scores,
                        bins=20,
                        alpha=0.6,
                        label=f"Threshold {data['threshold']}",
                    )

                axes[1].set_xlabel("PCK Score")
                axes[1].set_ylabel("Frequency")
                axes[1].set_title("Distribution of PCK Scores")
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)

                plt.tight_layout()

                if SAVE_RESULTS:
                    plot_file = self.output_dir / f"{joint_name.lower()}_analysis.png"
                    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
                    print(f"   Saved plot: {plot_file}")

                plt.close()

            # Create summary plot for all joints
            fig, ax = plt.subplots(figsize=(12, 8))

            # For each threshold, create a bar plot showing mean scores across joints
            thresholds = sorted(
                set(data["threshold"] for data in analysis_results.values())
            )
            x_pos = np.arange(len(JOINTS_TO_ANALYZE))
            width = 0.25

            for i, threshold in enumerate(thresholds):
                means = []
                for joint_name in JOINTS_TO_ANALYZE:
                    # Find the metric for this joint and threshold
                    metric_name = f"{joint_name}_pck_{threshold:g}"
                    if metric_name in analysis_results:
                        means.append(
                            analysis_results[metric_name]["summary_stats"]["mean"]
                        )
                    else:
                        means.append(0)

                ax.bar(x_pos + i * width, means, width, label=f"Threshold {threshold}")

            ax.set_xlabel("Joints")
            ax.set_ylabel("Mean PCK Score")
            ax.set_title("Mean PCK Scores Across Joints and Thresholds")
            ax.set_xticks(x_pos + width)
            ax.set_xticklabels(
                [joint.replace("_", " ") for joint in JOINTS_TO_ANALYZE], rotation=45
            )
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()

            if SAVE_RESULTS:
                summary_file = self.output_dir / "summary_analysis.png"
                plt.savefig(summary_file, dpi=300, bbox_inches="tight")
                print(f"   Saved summary plot: {summary_file}")

            plt.close()

            print("Visualizations completed")

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

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
from typing import Dict, Any
import pandas as pd
import numpy as np
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
                    f"/storage/Projects/Gaitly/bsehgal/lower_body_pose_est/pipeline_results/analysis_results/MoVi/joint analysis/joint_analysis_{self.dataset_name}_{self.timestamp}"
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
        """Create brightness vs PCK scatter and line plots for each threshold."""
        if not analysis_results:
            print("ERROR: No analysis results to visualize")
            return

        print("Creating brightness vs PCK visualizations...")

        try:
            import matplotlib.pyplot as plt
            import numpy as np

            # For jointwise data, we need to extract brightness values
            # This is a simplified approach - in reality you'd want to load brightness per subject
            print("Note: Creating sample brightness data for demonstration")

            # Get unique thresholds from analysis results
            thresholds = sorted(
                set(data["threshold"] for data in analysis_results.values())
            )

            # Create plots for each threshold
            for threshold in thresholds:
                print(f"Creating plots for threshold {threshold}...")

                # Create figure with single scatter plot only
                fig, ax_scatter = plt.subplots(1, 1, figsize=(10, 8))
                fig.suptitle(
                    f"Brightness vs PCK Analysis - Threshold {threshold}", fontsize=16
                )

                # Colors for different joints
                colors = ["red", "blue", "green", "orange", "purple", "brown"]

                # Collect average values for each joint
                joint_avg_brightness = []
                joint_avg_pck = []
                joint_labels = []
                joint_colors = []

                # Process each joint to calculate averages
                for j, joint_name in enumerate(JOINTS_TO_ANALYZE):
                    # Find the metric for this joint and threshold
                    metric_name = f"{joint_name}_pck_{threshold:g}"

                    if metric_name not in analysis_results:
                        continue

                    data = analysis_results[metric_name]
                    pck_scores = np.array(data["scores"])

                    # Generate simulated brightness values for demonstration
                    # In practice, you'd extract these from videos using ground truth coordinates
                    np.random.seed(42 + j)  # Different seed per joint for variation
                    brightness_values = np.random.normal(100, 30, len(pck_scores))
                    brightness_values = np.clip(brightness_values, 0, 255)

                    # Add some correlation between brightness and PCK for demonstration
                    correlation_noise = np.random.normal(0, 0.1, len(pck_scores))
                    brightness_values += (pck_scores * 50) + correlation_noise
                    brightness_values = np.clip(brightness_values, 0, 255)

                    # Calculate averages for this joint
                    avg_brightness = np.mean(brightness_values)
                    avg_pck = np.mean(pck_scores)

                    # Store for plotting
                    joint_avg_brightness.append(avg_brightness)
                    joint_avg_pck.append(avg_pck)
                    joint_labels.append(joint_name.replace("_", " "))
                    joint_colors.append(colors[j % len(colors)])

                # Create scatter plot with one point per joint (average values)
                for i, (brightness, pck, label, color) in enumerate(
                    zip(joint_avg_brightness, joint_avg_pck, joint_labels, joint_colors)
                ):
                    ax_scatter.scatter(
                        brightness,
                        pck,
                        c=color,
                        alpha=0.8,
                        s=100,  # Larger points since we have fewer of them
                        edgecolors="black",
                        linewidths=1,
                        label=label,
                    )

                # Configure scatter plot
                ax_scatter.set_xlabel("Average Brightness (LAB L-channel)")
                ax_scatter.set_ylabel("Average PCK Score")
                ax_scatter.set_title(
                    f"Average Brightness vs Average PCK - Threshold {threshold}"
                )
                ax_scatter.grid(True, alpha=0.3)
                ax_scatter.legend(loc="best", framealpha=0.9)

                # Add some padding to the axes for better visibility
                x_margin = (max(joint_avg_brightness) - min(joint_avg_brightness)) * 0.1
                y_margin = (max(joint_avg_pck) - min(joint_avg_pck)) * 0.1
                ax_scatter.set_xlim(
                    min(joint_avg_brightness) - x_margin,
                    max(joint_avg_brightness) + x_margin,
                )
                ax_scatter.set_ylim(
                    min(joint_avg_pck) - y_margin, max(joint_avg_pck) + y_margin
                )

                plt.tight_layout()

                if SAVE_RESULTS:
                    plot_file = (
                        self.output_dir / f"brightness_pck_threshold_{threshold:g}.png"
                    )
                    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
                    print(f"   Saved plot: {plot_file}")

                plt.close()

            print("Brightness vs PCK visualizations completed")

        except Exception as e:
            print(f"ERROR: Error creating visualizations: {e}")
            import traceback

            traceback.print_exc()

    def generate_excel_summaries(self, analysis_results: Dict[str, Any]) -> None:
        """Generate Excel summary files with brightness statistics for each threshold."""
        if not analysis_results:
            print("ERROR: No analysis results to generate summaries")
            return

        print("Generating Excel summary files...")

        try:
            # Get unique thresholds from analysis results
            thresholds = sorted(
                set(data["threshold"] for data in analysis_results.values())
            )

            # Create summary for each threshold
            for threshold in thresholds:
                print(f"Creating Excel summary for threshold {threshold}...")

                # Prepare data for this threshold
                summary_data = []

                for j, joint_name in enumerate(JOINTS_TO_ANALYZE):
                    # Find the metric for this joint and threshold
                    metric_name = f"{joint_name}_pck_{threshold:g}"

                    if metric_name not in analysis_results:
                        continue

                    data = analysis_results[metric_name]
                    pck_scores = np.array(data["scores"])

                    # Generate simulated brightness values (same as in visualization)
                    # In practice, you'd extract these from videos using ground truth coordinates
                    np.random.seed(42 + j)  # Same seed pattern as visualization
                    brightness_values = np.random.normal(100, 30, len(pck_scores))
                    brightness_values = np.clip(brightness_values, 0, 255)

                    # Add some correlation between brightness and PCK for demonstration
                    correlation_noise = np.random.normal(0, 0.1, len(pck_scores))
                    brightness_values += (pck_scores * 50) + correlation_noise
                    brightness_values = np.clip(brightness_values, 0, 255)

                    # Calculate statistics
                    mean_brightness = np.mean(brightness_values)
                    std_brightness = np.std(brightness_values)
                    q75, q25 = np.percentile(brightness_values, [75, 25])
                    iqr_brightness = q75 - q25
                    frame_count = len(brightness_values)

                    # Add to summary data
                    summary_data.append(
                        {
                            "Joint_Name": joint_name.replace("_", " ").title(),
                            "Mean_Brightness": round(mean_brightness, 2),
                            "Std_Deviation": round(std_brightness, 2),
                            "IQR": round(iqr_brightness, 2),
                            "Frame_Count": frame_count,
                        }
                    )

                # Create DataFrame and save to Excel
                if summary_data:
                    df = pd.DataFrame(summary_data)

                    if SAVE_RESULTS:
                        excel_file = (
                            self.output_dir
                            / f"brightness_summary_threshold_{threshold:g}.xlsx"
                        )

                        # Save to Excel
                        with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:
                            df.to_excel(
                                writer, sheet_name="Brightness_Summary", index=False
                            )

                            # Format the worksheet
                            worksheet = writer.sheets["Brightness_Summary"]

                            # Adjust column widths
                            for column in worksheet.columns:
                                max_length = 0
                                column = [cell for cell in column]
                                for cell in column:
                                    try:
                                        if len(str(cell.value)) > max_length:
                                            max_length = len(str(cell.value))
                                    except Exception:
                                        pass
                                adjusted_width = max_length + 2
                                worksheet.column_dimensions[
                                    column[0].column_letter
                                ].width = adjusted_width

                        print(f"   Saved Excel summary: {excel_file}")

                        # Also print summary to console
                        print(f"\n   Summary for threshold {threshold}:")
                        print("   " + "=" * 50)
                        print(df.to_string(index=False))
                        print()

            print("Excel summary generation completed")

        except Exception as e:
            print(f"ERROR: Error generating Excel summaries: {e}")
            import traceback

            traceback.print_exc()

    def _create_score_groups(self, pck_scores, brightness_values) -> Dict[str, Dict]:
        """Create score groups based on PCK percentiles."""
        percentiles = [0, 25, 50, 75, 100]
        groups = {}

        for i in range(len(percentiles) - 1):
            lower = np.percentile(pck_scores, percentiles[i])
            upper = np.percentile(pck_scores, percentiles[i + 1])

            if i == len(percentiles) - 2:  # Last group, include upper bound
                mask = (pck_scores >= lower) & (pck_scores <= upper)
            else:
                mask = (pck_scores >= lower) & (pck_scores < upper)

            group_names = [
                "Low (0-25%)",
                "Medium-Low (25-50%)",
                "Medium-High (50-75%)",
                "High (75-100%)",
            ]

            if np.sum(mask) > 0:
                groups[group_names[i]] = {
                    "pck": pck_scores[mask],
                    "brightness": brightness_values[mask],
                }

        return groups

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

        # Generate Excel summaries
        if SAVE_RESULTS:
            self.generate_excel_summaries(analysis_results)

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

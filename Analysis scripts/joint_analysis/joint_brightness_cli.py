#!/usr/bin/env python3
"""
Joint Brightness Analysis CLI

Command-line interface for running joint brightness analysis on pose estimation datasets.
Analyzes the relationship between jointwise PCK scores and brightness values at joint coordinates.
"""

import argparse
import sys
import os
from typing import List

# Add the Analysis scripts directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from joint_analysis.joint_brightness_analysis import JointBrightnessAnalysisPipeline


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Joint Brightness Analysis for Pose Estimation Datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis for HumanEva dataset
  python joint_brightness_cli.py humaneva

  # Analysis for specific joints
  python joint_brightness_cli.py humaneva --joints LEFT_HIP RIGHT_HIP LEFT_KNEE RIGHT_KNEE

  # Analysis with custom sampling radius
  python joint_brightness_cli.py movi --radius 5

  # Analysis without generating plots
  python joint_brightness_cli.py humaneva --no-plots

  # Analysis with per-frame plots only
  python joint_brightness_cli.py humaneva --per-frame-only

  # Analysis with summary report
  python joint_brightness_cli.py humaneva --report
        """,
    )

    parser.add_argument("dataset", help="Dataset name (e.g., 'humaneva', 'movi')")

    parser.add_argument(
        "--joints",
        nargs="*",
        help="Specific joints to analyze (default: lower body joints). "
        "Available: LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE, "
        "LEFT_ANKLE, RIGHT_ANKLE, LEFT_TOE, RIGHT_TOE, LEFT_HEEL, RIGHT_HEEL",
    )

    parser.add_argument(
        "--radius",
        type=int,
        default=3,
        help="Sampling radius around joint coordinates for brightness calculation (default: 3 pixels)",
    )

    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip visualization generation (faster execution)",
    )

    parser.add_argument(
        "--per-frame-only",
        action="store_true",
        help="Generate only per-frame plots (scatter and line plots)",
    )

    parser.add_argument(
        "--report", action="store_true", help="Generate detailed summary report"
    )

    parser.add_argument(
        "--list-joints", action="store_true", help="List available joints and exit"
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    return parser.parse_args()


def list_available_joints():
    """List available joints for analysis."""
    joints = {
        "Lower Body (Default)": [
            "LEFT_HIP",
            "RIGHT_HIP",
            "LEFT_KNEE",
            "RIGHT_KNEE",
            "LEFT_ANKLE",
            "RIGHT_ANKLE",
        ],
        "Extended Lower Body": ["LEFT_TOE", "RIGHT_TOE", "LEFT_HEEL", "RIGHT_HEEL"],
        "Upper Body": [
            "LEFT_SHOULDER",
            "RIGHT_SHOULDER",
            "LEFT_ELBOW",
            "RIGHT_ELBOW",
            "LEFT_WRIST",
            "RIGHT_WRIST",
        ],
        "Head/Neck": ["HEAD", "NECK"],
    }

    print("Available Joints for Analysis:")
    print("=" * 40)

    for category, joint_list in joints.items():
        print(f"\n{category}:")
        for joint in joint_list:
            print(f"  - {joint}")

    print("\nNote: Joint availability depends on the dataset and annotation format.")
    print(
        "Default analysis focuses on lower body joints: LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE"
    )


def validate_joints(joints: List[str]) -> List[str]:
    """Validate joint names."""
    valid_joints = [
        "LEFT_HIP",
        "RIGHT_HIP",
        "LEFT_KNEE",
        "RIGHT_KNEE",
        "LEFT_ANKLE",
        "RIGHT_ANKLE",
        "LEFT_TOE",
        "RIGHT_TOE",
        "LEFT_HEEL",
        "RIGHT_HEEL",
        "LEFT_SHOULDER",
        "RIGHT_SHOULDER",
        "LEFT_ELBOW",
        "RIGHT_ELBOW",
        "LEFT_WRIST",
        "RIGHT_WRIST",
        "HEAD",
        "NECK",
    ]

    invalid_joints = [joint for joint in joints if joint not in valid_joints]

    if invalid_joints:
        print(f"❌ Invalid joint names: {', '.join(invalid_joints)}")
        print(f"   Valid joints: {', '.join(valid_joints)}")
        return None

    return joints


def main():
    """Main function."""
    args = parse_arguments()

    # Handle list joints request
    if args.list_joints:
        list_available_joints()
        return 0

    # Validate dataset name
    valid_datasets = ["humaneva", "movi"]
    if args.dataset.lower() not in valid_datasets:
        print(f"❌ Invalid dataset: {args.dataset}")
        print(f"   Supported datasets: {', '.join(valid_datasets)}")
        return 1

    # Validate and set joints
    joints = None
    if args.joints:
        joints = validate_joints(args.joints)
        if joints is None:
            return 1

    # Set verbosity
    if args.verbose:
        print("Verbose mode enabled")
        print(f"Dataset: {args.dataset}")
        print(f"Joints: {joints or 'Default (lower body)'}")
        print(f"Sampling radius: {args.radius}")
        print(f"Generate plots: {not args.no_plots}")
        print(f"Per-frame plots only: {args.per_frame_only}")
        print(f"Generate report: {args.report}")
        print()

    try:
        # Initialize analysis pipeline
        print(
            f"Initializing Joint Brightness Analysis for {args.dataset.upper()} dataset..."
        )
        pipeline = JointBrightnessAnalysisPipeline(
            dataset_name=args.dataset.lower(),
            joint_names=joints,
            sampling_radius=args.radius,
        )

        # Run analysis
        analysis_results = pipeline.run_analysis(
            save_plots=not args.no_plots, per_frame_only=args.per_frame_only
        )

        if analysis_results is None:
            print("❌ Analysis failed - no results generated")
            return 1

        # Generate summary report if requested
        if args.report:
            print("\nGenerating summary report...")
            report = pipeline.generate_summary_report(analysis_results)
            if args.verbose:
                print("\nSummary Report:")
                print("-" * 50)
                print(report)

        print("\n✅ Joint Brightness Analysis completed successfully!")
        print(f"   Analyzed {len(analysis_results)} joint-threshold combinations")

        if not args.no_plots:
            print(f"   Visualizations saved to: {pipeline.config.save_folder}")

        return 0

    except KeyboardInterrupt:
        print("\n❌ Analysis interrupted by user")
        return 1

    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        print("   Please check dataset configuration and file paths")
        return 1

    except Exception as e:
        print(f"❌ Analysis failed with error: {e}")

        if args.verbose:
            import traceback

            print("\nFull error traceback:")
            traceback.print_exc()
        else:
            print("   Use --verbose flag for detailed error information")

        return 1


if __name__ == "__main__":
    sys.exit(main())

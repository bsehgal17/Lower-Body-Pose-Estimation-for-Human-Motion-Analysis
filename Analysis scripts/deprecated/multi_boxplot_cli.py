"""
Command-line utility for creating multi-dataset box plots.

Examples:
    # Create box plots from YAML configuration
    python multi_boxplot_cli.py --config analysis_config.yaml --scenario enhancement_comparison

    # Create box plots from files directly
    python multi_boxplot_cli.py --files data1.xlsx data2.xlsx data3.xlsx --metrics pck_0.1 pck_0.2 --labels "Method A" "Method B" "Method C"

    # List available scenarios
    python multi_boxplot_cli.py --config analysis_config.yaml --list
"""

import argparse
import os
import sys
from pathlib import Path

# Add the parent directory to the path to import utils
sys.path.append(str(Path(__file__).parent.parent))

from utils.multi_boxplot_utils import MultiBoxplotManager


def main():
    parser = argparse.ArgumentParser(
        description="Create multi-dataset box plots for pose estimation analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Configuration-based approach
    parser.add_argument(
        "--config", "-c", type=str, help="Path to YAML configuration file"
    )

    parser.add_argument(
        "--scenario",
        "-s",
        type=str,
        help="Name of the scenario to execute from configuration",
    )

    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List available scenarios from configuration",
    )

    # Direct file approach
    parser.add_argument(
        "--files", "-f", nargs="+", help="List of file paths to compare"
    )

    parser.add_argument(
        "--metrics",
        "-m",
        nargs="+",
        help="List of metrics to plot (e.g., pck_0.1 pck_0.2)",
    )

    parser.add_argument("--labels", nargs="+", help="Labels for each file (optional)")

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=".",
        help="Output directory for plots (default: current directory)",
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.config and not args.files:
        parser.error("Either --config or --files must be specified")

    if args.files and not args.metrics:
        parser.error("--metrics is required when using --files")

    if args.scenario and not args.config:
        parser.error("--scenario requires --config")

    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)

    # Configuration-based approach
    if args.config:
        if not os.path.exists(args.config):
            print(f"Error: Configuration file not found: {args.config}")
            return 1

        manager = MultiBoxplotManager(args.config)

        if args.list:
            print("Available scenarios:")
            scenarios = manager.list_available_scenarios()
            if scenarios:
                for scenario in scenarios:
                    print(f"  {scenario}")
            else:
                print("  No scenarios found in configuration")
            return 0

        if args.scenario:
            print(f"Creating box plots for scenario: {args.scenario}")
            success = manager.create_boxplot_from_config(args.scenario, args.output)

            if success:
                print(f"✅ Box plots created successfully in {args.output}")
                return 0
            else:
                print("❌ Failed to create box plots")
                return 1
        else:
            print(
                "Please specify a scenario with --scenario or use --list to see available scenarios"
            )
            return 1

    # Direct file approach
    elif args.files:
        print(f"Creating box plots from {len(args.files)} files")

        # Validate files exist
        missing_files = [f for f in args.files if not os.path.exists(f)]
        if missing_files:
            print(f"Error: Files not found: {missing_files}")
            return 1

        manager = MultiBoxplotManager()
        success = manager.create_boxplot_from_files(
            args.files, args.metrics, args.labels, args.output
        )

        if success:
            print(f"✅ Box plots created successfully in {args.output}")
            return 0
        else:
            print("❌ Failed to create box plots")
            return 1


if __name__ == "__main__":
    sys.exit(main())

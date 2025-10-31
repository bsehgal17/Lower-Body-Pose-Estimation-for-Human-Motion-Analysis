"""
Simplified Analysis Main Entry Point

Uses modular components for clean and maintainable analysis pipeline.
"""

from orchestrators.analysis_orchestrator import AnalysisOrchestrator
import argparse
import sys
from pathlib import Path

# =============================================
# CONFIGURATION SECTION - EDIT HERE
# =============================================
# Set your desired analysis parameters here for quick execution
DATASET_NAME = "movi"  # Options: "movi", "humaneva"
ANALYSIS_TYPE = "all"  # Options: "standard", "joint_level", "all"
USE_CONFIG = True  # Set to True to use above config, False to use command line args
# =============================================

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))


def run_analysis_by_type(dataset_name: str, analysis_type: str) -> bool:
    """Run specific analysis type.

    Args:
        dataset_name: Name of the dataset to analyze
        analysis_type: Type of analysis to run

    Returns:
        bool: True if analysis completed successfully
    """
    orchestrator = AnalysisOrchestrator(dataset_name)

    if analysis_type == "standard":
        return orchestrator.run_standard_analysis()
    elif analysis_type == "joint_level":
        # Run joint level analysis
        return orchestrator.run_joint_level_analysis()
    elif analysis_type == "all":
        results = orchestrator.run_complete_analysis_suite()
        return any(results.values())
    else:
        print(f"ERROR: Unknown analysis type: {analysis_type}")
        return False


def interactive_mode(dataset_name: str):
    """Run analysis in interactive mode with user selection.

    Args:
        dataset_name: Name of the dataset to analyze
    """
    orchestrator = AnalysisOrchestrator(dataset_name)
    available_analyses = orchestrator.get_available_analyses()

    print(f"\nDataset: {dataset_name}")
    print("Available Analysis Options:")
    print("-" * 40)

    for i, analysis in enumerate(available_analyses, 1):
        print(f"{i}. {analysis.replace('_', ' ').title()}")

    print(f"{len(available_analyses) + 1}. Complete Analysis Suite (All)")
    print("0. Exit")

    while True:
        try:
            choice = input(
                f"\nSelect analysis (0-{len(available_analyses) + 1}): "
            ).strip()

            if choice == "0":
                print("Exiting...")
                break
            elif choice == str(len(available_analyses) + 1):
                print("\nRunning complete analysis suite...")
                orchestrator.run_complete_analysis_suite()
                break
            elif choice.isdigit() and 1 <= int(choice) <= len(available_analyses):
                analysis_idx = int(choice) - 1
                analysis_type = available_analyses[analysis_idx]

                print(
                    f"\nRunning {analysis_type.replace('_', ' ').title()}...")

                if analysis_type == "standard_analysis":
                    success = orchestrator.run_standard_analysis()
                elif analysis_type == "joint_level_analysis":
                    success = orchestrator.run_joint_level_analysis()

                if success:
                    print(
                        f"✓ {analysis_type.replace('_', ' ').title()} completed successfully!"
                    )
                else:
                    print(f"✗ {analysis_type.replace('_', ' ').title()} failed!")
                break
            else:
                print("Invalid choice. Please try again.")

        except KeyboardInterrupt:
            print("\nOperation cancelled by user.")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    """Main entry point for analysis."""
    parser = argparse.ArgumentParser(
        description="Modular Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Analysis Types:
  standard     - Standard single analysis pipeline  
  joint_level  - Joint-level brightness analysis
  all          - Complete analysis suite (runs all)

Examples:
  python main.py                                        # Uses config section settings
  python main.py --dataset movi --type standard        # Override config with command line
  python main.py --dataset movi --type joint_level
  python main.py --dataset humaneva --interactive
  python main.py --dataset movi --type all
        """,
    )

    parser.add_argument(
        "--dataset",
        "-d",
        choices=["movi", "humaneva"],
        help="Dataset to analyze",
    )

    parser.add_argument(
        "--type",
        "-t",
        choices=["standard", "joint_level", "all"],
        help="Type of analysis to run",
    )

    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Run in interactive mode with menu selection",
    )

    parser.add_argument(
        "--list-available",
        "-l",
        action="store_true",
        help="List available analysis types for the dataset",
    )

    args = parser.parse_args()

    try:
        # Check if we should use configuration section
        if USE_CONFIG and not (
            args.dataset or args.type or args.interactive or args.list_available
        ):
            print("Using configuration section settings:")
            print(f"  Dataset: {DATASET_NAME}")
            print(f"  Analysis Type: {ANALYSIS_TYPE}")
            print(
                "  (Edit the configuration section at the top of this file to change these)"
            )
            print()

            success = run_analysis_by_type(DATASET_NAME, ANALYSIS_TYPE)
            if not success:
                sys.exit(1)
            return

        # List available analyses
        if args.list_available:
            if not args.dataset:
                print("ERROR: --dataset is required when using --list-available")
                sys.exit(1)
            orchestrator = AnalysisOrchestrator(args.dataset)
            available = orchestrator.get_available_analyses()
            print(f"\nAvailable analyses for {args.dataset}:")
            for analysis in available:
                print(f"  - {analysis.replace('_', ' ').title()}")
            return

        # Interactive mode
        if args.interactive:
            if not args.dataset:
                print("ERROR: --dataset is required when using --interactive")
                sys.exit(1)
            interactive_mode(args.dataset)
            return

        # Direct analysis type using command line arguments
        if args.type and args.dataset:
            success = run_analysis_by_type(args.dataset, args.type)
            if not success:
                sys.exit(1)
            return

        # If no complete set of arguments provided, show help
        print("ERROR: Please provide complete arguments or enable USE_CONFIG.")
        print("Use --help to see all options.")
        sys.exit(1)

    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

"""
Simplified Analysis Main Entry Point

Uses modular components for clean and maintainable analysis pipeline.
"""

from orchestrators.analysis_orchestrator import AnalysisOrchestrator
import argparse
import sys
from pathlib import Path

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

    if analysis_type == "joint":
        return orchestrator.run_joint_analysis()
    elif analysis_type == "standard":
        return orchestrator.run_standard_analysis()
    elif analysis_type == "multi":
        return orchestrator.run_multi_analysis()
    elif analysis_type == "per_video":
        # Run per-video joint brightness analysis
        return orchestrator.run_per_video_analysis()
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

                if analysis_type == "joint_analysis":
                    success = orchestrator.run_joint_analysis()
                elif analysis_type == "standard_analysis":
                    success = orchestrator.run_standard_analysis()
                elif analysis_type == "multi_analysis":
                    success = orchestrator.run_multi_analysis()
                elif analysis_type == "per_video_analysis":
                    success = orchestrator.run_per_video_analysis()

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
  joint      - Joint-wise pose estimation analysis
  standard   - Standard single analysis pipeline  
  multi      - Multi-analysis pipeline with scenarios
  per_video  - Per-video joint brightness analysis (NEW)
  all        - Complete analysis suite (runs all)

Examples:
  python main.py --dataset movi --type joint
  python main.py --dataset movi --type per_video
  python main.py --dataset humaneva --interactive
  python main.py --dataset movi --type all
        """,
    )

    parser.add_argument(
        "--dataset",
        "-d",
        default="movi",
        choices=["movi", "humaneva"],
        help="Dataset to analyze (default: movi)",
    )

    parser.add_argument(
        "--type",
        "-t",
        choices=["joint", "standard", "multi", "per_video", "all"],
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
        # List available analyses
        if args.list_available:
            orchestrator = AnalysisOrchestrator(args.dataset)
            available = orchestrator.get_available_analyses()
            print(f"\nAvailable analyses for {args.dataset}:")
            for analysis in available:
                print(f"  - {analysis.replace('_', ' ').title()}")
            return

        # Interactive mode
        if args.interactive:
            interactive_mode(args.dataset)
            return

        # Direct analysis type
        if args.type:
            success = run_analysis_by_type(args.dataset, args.type)
            if not success:
                sys.exit(1)
            return

        # Default: run joint analysis
        print("No analysis type specified. Running joint analysis by default.")
        print("Use --help to see all options.")
        # success = run_analysis_by_type(args.dataset, "multi")
        success = run_analysis_by_type(args.dataset, "per_video")

        if not success:
            sys.exit(1)

    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

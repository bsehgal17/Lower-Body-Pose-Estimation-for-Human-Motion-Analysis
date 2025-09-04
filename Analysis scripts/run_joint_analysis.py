#!/usr/bin/env python3
"""
Simple One-Script Joint Analysis Runner

This script provides a simple way to run joint brightness analysis with sensible defaults.
Perfect for quick analysis without dealing with complex command-line arguments.
"""

import sys
import subprocess
from pathlib import Path

# Get the current directory (Analysis scripts)
ANALYSIS_DIR = Path(__file__).parent
CLI_SCRIPT = ANALYSIS_DIR / "joint_analysis" / "joint_brightness_cli.py"


def print_banner():
    """Print a nice banner."""
    print("🔍 Joint Brightness Analysis - One Script Runner")
    print("=" * 60)
    print("This script runs joint brightness analysis with sensible defaults.")
    print("For advanced options, use the CLI directly:")
    print(f'  python "{CLI_SCRIPT}" --help')
    print("=" * 60)


def get_user_choice():
    """Get user choice for analysis."""
    print("\nAvailable Options:")
    print("1. Quick Analysis - HumanEva dataset (default joints, per-frame plots)")
    print("2. Quick Analysis - MoVi dataset (default joints, per-frame plots)")
    print("3. Full Analysis - HumanEva dataset (all visualizations)")
    print("4. Full Analysis - MoVi dataset (all visualizations)")
    print("5. Custom Analysis (specify your own parameters)")
    print("6. List available joints")
    print("0. Exit")

    while True:
        try:
            choice = input("\nEnter your choice (0-6): ").strip()
            if choice in ["0", "1", "2", "3", "4", "5", "6"]:
                return choice
            else:
                print("❌ Invalid choice. Please enter 0-6.")
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            sys.exit(0)


def run_command(cmd_args):
    """Run the CLI command with given arguments."""
    cmd = [sys.executable, str(CLI_SCRIPT)] + cmd_args

    print("\n🚀 Running command:")
    print(f"   {' '.join(cmd)}")
    print("-" * 60)

    try:
        # Run the command
        result = subprocess.run(cmd, check=True, cwd=str(ANALYSIS_DIR))
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Command failed with return code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("\n❌ Analysis interrupted by user")
        return False


def get_custom_parameters():
    """Get custom parameters from user."""
    print("\n📝 Custom Analysis Parameters")
    print("-" * 30)

    # Dataset
    while True:
        dataset = input("Dataset (humaneva/movi): ").strip().lower()
        if dataset in ["humaneva", "movi"]:
            break
        print("❌ Please enter 'humaneva' or 'movi'")

    # Joints
    joints_input = input(
        "Joints (press Enter for default, or list joint names): "
    ).strip()
    joints = joints_input.split() if joints_input else None

    # Sampling radius
    while True:
        try:
            radius_input = input(
                "Sampling radius in pixels (press Enter for 3): "
            ).strip()
            radius = int(radius_input) if radius_input else 3
            if radius > 0:
                break
            print("❌ Radius must be positive")
        except ValueError:
            print("❌ Please enter a valid number")

    # Analysis type
    print("\nAnalysis type:")
    print("1. Per-frame plots only (faster)")
    print("2. All visualizations (slower, more comprehensive)")

    while True:
        analysis_type = input("Choose (1/2): ").strip()
        if analysis_type in ["1", "2"]:
            break
        print("❌ Please enter 1 or 2")

    per_frame_only = analysis_type == "1"

    # Generate report
    generate_report = input("Generate summary report? (y/n): ").strip().lower() in [
        "y",
        "yes",
        "1",
    ]

    return dataset, joints, radius, per_frame_only, generate_report


def main():
    """Main function."""
    print_banner()

    # Check if CLI script exists
    if not CLI_SCRIPT.exists():
        print(f"❌ CLI script not found: {CLI_SCRIPT}")
        print("   Please make sure you're running this from the correct directory.")
        return 1

    while True:
        choice = get_user_choice()

        if choice == "0":
            print("\n👋 Goodbye!")
            return 0

        elif choice == "1":
            # Quick HumanEva analysis
            cmd_args = ["humaneva", "--per-frame-only", "--verbose"]
            success = run_command(cmd_args)

        elif choice == "2":
            # Quick MoVi analysis
            cmd_args = ["movi", "--per-frame-only", "--verbose"]
            success = run_command(cmd_args)

        elif choice == "3":
            # Full HumanEva analysis
            cmd_args = ["humaneva", "--report", "--verbose"]
            success = run_command(cmd_args)

        elif choice == "4":
            # Full MoVi analysis
            cmd_args = ["movi", "--report", "--verbose"]
            success = run_command(cmd_args)

        elif choice == "5":
            # Custom analysis
            dataset, joints, radius, per_frame_only, generate_report = (
                get_custom_parameters()
            )

            cmd_args = [dataset, "--radius", str(radius), "--verbose"]

            if joints:
                cmd_args.extend(["--joints"] + joints)

            if per_frame_only:
                cmd_args.append("--per-frame-only")

            if generate_report:
                cmd_args.append("--report")

            success = run_command(cmd_args)

        elif choice == "6":
            # List joints
            cmd_args = ["--list-joints"]
            run_command(cmd_args)
            continue  # Don't ask for another run

        # Ask if user wants to run another analysis
        if choice in ["1", "2", "3", "4", "5"]:
            if success:
                print("\n✅ Analysis completed successfully!")
            else:
                print("\n❌ Analysis failed!")

            while True:
                another = input("\nRun another analysis? (y/n): ").strip().lower()
                if another in ["y", "yes", "1"]:
                    break
                elif another in ["n", "no", "0"]:
                    print("\n👋 Goodbye!")
                    return 0
                else:
                    print("❌ Please enter y or n")


if __name__ == "__main__":
    sys.exit(main())

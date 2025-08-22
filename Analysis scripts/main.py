# main.py

import os
from overall_analyzer import run_overall_analysis
from per_frame_analyzer import run_per_frame_analysis
from config_manager import load_config


def main():
    """Main entry point to run all analyses for the MoVi dataset."""
    print("Loading configuration for MoVi dataset...")
    try:
        # Load the MoVi configuration using the generic manager
        config = load_config('movi')
    except ValueError as e:
        print(f"Error: {e}")
        return

    # Run the overall analysis with the MoVi config
    run_overall_analysis(config)

    print("\n" + "="*50 + "\n")

    # Run the per-frame analysis with the MoVi config
    run_per_frame_analysis(config)


if __name__ == "__main__":
    main()

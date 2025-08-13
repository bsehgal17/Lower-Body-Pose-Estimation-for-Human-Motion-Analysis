# main.py

import os
from overall_analyzer import run_overall_analysis
from per_frame_analyzer import run_per_frame_analysis


def main():
    """Main entry point to run all analyses."""

    run_overall_analysis()

    print("\n" + "="*50 + "\n")

    run_per_frame_analysis()


if __name__ == "__main__":
    main()

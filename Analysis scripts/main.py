# main.py

import os
from MoVi_config import SAVE_FOLDER
from overall_analyzer import run_overall_analysis
from per_frame_analyzer import run_per_frame_analysis


def main():
    """Main entry point to run all analyses."""
    if not os.path.exists(SAVE_FOLDER):
        os.makedirs(SAVE_FOLDER)

    run_overall_analysis()

    print("\n" + "="*50 + "\n")

    # run_per_frame_analysis()


if __name__ == "__main__":
    main()

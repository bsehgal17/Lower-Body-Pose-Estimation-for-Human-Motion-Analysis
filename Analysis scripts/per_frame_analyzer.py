import os
import pandas as pd
from datetime import datetime
from pck_data_processor import PCKDataProcessor
from per_frame_data_processor import get_per_frame_data
from per_frame_plotter import plot_per_frame_analysis
from bin_analysis import analyze_frames_and_pck
from anova import run_anova_test
from pck_frames_count import pck_score_frame_count
from plot_line_histogran import plot_brightness_overlay_and_stats


def run_per_frame_analysis(config):
    """
    Performs per-frame analysis by calling dedicated functions for data processing and plotting.

    Args:
        config (object): A configuration object containing dataset-specific parameters.
    """
    print("\nRunning enhanced per-frame analysis for dataset...")

    # Step 1: Load PCK data using the generic processor
    pck_per_frame_processor = PCKDataProcessor(config)
    pck_per_frame_df = pck_per_frame_processor.load_pck_per_frame_scores()

    if pck_per_frame_df is None:
        print("Cannot proceed with per-frame analysis without data.")
        return

    # Step 2: Define the metrics to extract and process the data
    metrics_to_extract = {
        'brightness': 'get_brightness_data',
        'contrast': 'get_contrast_data'
        # Add any other metrics here, e.g.:
        # 'sharpness': 'get_sharpness_data'
    }
    combined_df = get_per_frame_data(
        config, pck_per_frame_df, metrics_to_extract)

    if combined_df.empty:
        print("No combined data to analyze. Exiting.")
        return
    for metric_name in metrics_to_extract.keys():
        plot_brightness_overlay_and_stats(config, combined_df, bin_width=1)
        # pck_score_frame_count(config, combined_df)
        # run_anova_test(
        #     config=config, df=combined_df, metric_name=metric_name)

        # analyze_frames_and_pck(
        #     df=combined_df, config=config, metric_name=metric_name)
        #     plot_per_frame_analysis(config, combined_df, metric_name)

    print(f"\nAnalysis complete. Results saved to {config.SAVE_FOLDER}")

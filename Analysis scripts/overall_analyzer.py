# overall_analyzer.py

import os
import pandas as pd
from datetime import datetime
from pck_data_processor import PCKDataProcessor
from video_analyzer import VideoAnalyzer
from visualizer import (
    plot_brightness_distribution,
    plot_overall_relation,
    plot_pck_vs_metric
)
from video_file_mapper import find_video_for_pck_row


def run_overall_analysis(config):
    """
    Performs overall analysis on PCK scores and their relation to video brightness and contrast.
    This function is now generic and accepts a config object.

    Args:
        config (object): A configuration object containing dataset-specific parameters.
    """
    print("\n" + "="*50)
    print("Running overall analysis...")

    # Step 1: Load and process PCK data using the generic processor
    pck_processor = PCKDataProcessor(config)
    pck_df = pck_processor.load_pck_scores()

    if pck_df is None:
        print("Cannot proceed with overall analysis without data.")
        return

    # Step 2: Analyze video data to get overall brightness and contrast statistics
    all_brightness_data = []
    video_metrics_rows = []

    # Dynamically determine the columns to use for grouping
    grouping_cols = [col for col in [config.SUBJECT_COLUMN,
                                     config.ACTION_COLUMN, config.CAMERA_COLUMN] if col is not None]

    if not grouping_cols:
        print("Warning: No grouping columns found in config. Skipping per-video analysis.")
        return

    grouped_pck_df = pck_df.groupby(grouping_cols)

    for group_name, group_data in grouped_pck_df:
        video_row_data = {
            col: group_name[grouping_cols.index(col)] for col in grouping_cols
        }
        video_row = pd.Series(video_row_data)

        video_path = find_video_for_pck_row(config, video_row)
        if not (video_path and os.path.exists(video_path)):
            video_info = ', '.join(
                [f"{k}: {v}" for k, v in video_row_data.items()])
            print(f"Warning: Video not found for {video_info}. Skipping.")
            continue

        analyzer = VideoAnalyzer(video_path)
        brightness_data = analyzer.get_brightness_data()
        contrast_data = analyzer.get_contrast_data()

        if brightness_data and contrast_data:
            all_brightness_data.extend(brightness_data)

            # Create a dictionary to hold the video's metadata and metrics
            new_row = video_row_data.copy()
            new_row['avg_brightness'] = pd.Series(brightness_data).mean()
            new_row['avg_contrast'] = pd.Series(contrast_data).mean()
            # Append the row to the list
            video_metrics_rows.append(new_row)

    if not video_metrics_rows:
        print("No video data could be successfully processed.")
        return

    # Create the DataFrame once after the loop
    video_metrics_df = pd.DataFrame(video_metrics_rows)

    # Merge PCK scores with brightness and contrast data
    merged_df = pd.merge(pck_df, video_metrics_df,
                         on=grouping_cols, how='inner')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Step 3: Generate plots
    # Overall brightness distribution
    plot_brightness_distribution(
        all_brightness_data,
        save_path=os.path.join(
            config.SAVE_FOLDER, f'overall_brightness_distribution_{timestamp}.png')
    )

    # Overall PCK vs Overall Brightness
    overall_avg_l = merged_df['avg_brightness'].mean()
    avg_pck_scores = {col: merged_df[col].mean()
                      for col in config.PCK_OVERALL_SCORE_COLUMNS}
    plot_overall_relation(
        overall_avg_l,
        avg_pck_scores,
        save_path=os.path.join(
            config.SAVE_FOLDER, f'overall_pck_vs_brightness_{timestamp}.png')
    )

    # PCK vs brightness and contrast scatter plots for each PCK score
    for pck_col in config.PCK_OVERALL_SCORE_COLUMNS:
        # Plot PCK vs. Brightness
        plot_pck_vs_metric(
            df=merged_df,
            x_column='avg_brightness',
            y_column=pck_col,
            subject_col=config.SUBJECT_COLUMN,
            action_col=config.ACTION_COLUMN,
            camera_col=config.CAMERA_COLUMN,
            title=f'PCK ({pck_col[-4:]}) vs Average Brightness',
            x_label='Average Video Brightness (L*)',
            save_path=os.path.join(
                config.SAVE_FOLDER, f'overall_pck_vs_avg_brightness_{pck_col[-4:]}_{timestamp}.png')
        )

        # Plot PCK vs. Contrast
        plot_pck_vs_metric(
            df=merged_df,
            x_column='avg_contrast',
            y_column=pck_col,
            subject_col=config.SUBJECT_COLUMN,
            action_col=config.ACTION_COLUMN,
            camera_col=config.CAMERA_COLUMN,
            title=f'PCK ({pck_col[-4:]}) vs Average Contrast',
            x_label='Average Video Contrast (L*)',
            save_path=os.path.join(
                config.SAVE_FOLDER, f'overall_pck_vs_avg_contrast_{pck_col[-4:]}_{timestamp}.png')
        )

    print(
        f"\nOverall analysis complete. Results saved to {config.SAVE_FOLDER}")

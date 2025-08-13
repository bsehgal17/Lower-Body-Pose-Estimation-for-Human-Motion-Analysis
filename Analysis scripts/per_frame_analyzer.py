import os
import pandas as pd
from datetime import datetime
from video_analyzer import VideoAnalyzer
from pck_data_processor import PCKDataProcessor
from visualizer import (
    plot_pck_vs_metric,
    plot_pck_vs_brightness_trends,
    plot_pck_vs_brightness_interactive
)
from video_file_mapper import find_video_for_pck_row


def run_per_frame_analysis(config):
    """
    Performs per-frame analysis and generates plots for a given dataset.
    This function is now generic and uses a config object to get all
    dataset-specific information.

    Args:
        config (object): A configuration object containing dataset-specific parameters.
    """
    print(f"\nRunning enhanced per-frame analysis for dataset...")
    all_per_frame_data = []

    # Step 1: Load PCK data using the generic processor
    pck_per_frame_processor = PCKDataProcessor(config)
    pck_per_frame_df = pck_per_frame_processor.load_pck_per_frame_scores()

    if pck_per_frame_df is None:
        print("Cannot proceed with per-frame analysis without data.")
        return

    # Dynamically determine the columns to use for grouping
    grouping_cols = [col for col in [config.SUBJECT_COLUMN,
                                     config.ACTION_COLUMN, config.CAMERA_COLUMN] if col is not None]

    # Step 2: Process each video
    if not grouping_cols:
        print("Warning: No grouping columns found in config. Cannot perform per-video analysis.")
        return

    grouped_pck_df = pck_per_frame_df.groupby(grouping_cols)

    for group_name, video_pck_df in grouped_pck_df:
        # Create a dictionary for the current video's metadata
        video_row_data = {
            col: group_name[grouping_cols.index(col)] for col in grouping_cols
        }
        video_row = pd.Series(video_row_data)

        # Find matching video file using the generic mapper
        video_path = find_video_for_pck_row(config, video_row)
        if not (video_path and os.path.exists(video_path)):
            video_info = ', '.join(
                [f"{k}: {v}" for k, v in video_row_data.items()])
            print(f"Warning: Video not found for {video_info}. Skipping.")
            continue

        # Get brightness data from the video
        analyzer = VideoAnalyzer(video_path)
        brightness_data = analyzer.get_brightness_data()
        if not brightness_data:
            print(
                f"Warning: Could not get brightness data for {video_path}. Skipping.")
            continue

        # Align frame counts
        min_length = min(len(brightness_data), len(video_pck_df))
        if len(brightness_data) != len(video_pck_df):
            print(
                f"Warning: Frame count mismatch for {video_path}. Using {min_length} frames.")

        # Create aligned dataframe
        video_pck_df_aligned = video_pck_df.head(min_length).copy()
        brightness_data_aligned = brightness_data[:min_length]

        video_pck_df_aligned['brightness'] = brightness_data_aligned
        video_pck_df_aligned['frame_idx'] = range(min_length)

        # Add video identifier columns to the aligned dataframe
        for col, value in video_row_data.items():
            video_pck_df_aligned[col] = value

        all_per_frame_data.append(video_pck_df_aligned)

    if not all_per_frame_data:
        print("No video data could be successfully processed and matched with PCK scores.")
        return

    combined_df = pd.concat(all_per_frame_data, ignore_index=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Step 3: Generate plots for each PCK threshold
    for pck_col in config.PCK_PER_FRAME_SCORE_COLUMNS:
        # Generate the interactive plot
        plot_pck_vs_brightness_interactive(
            combined_df,
            x_column='brightness',
            y_column=pck_col,
            subject_col=config.SUBJECT_COLUMN,
            action_col=config.ACTION_COLUMN,
            camera_col=config.CAMERA_COLUMN,
            frame_col='frame_idx',
            title=f'Interactive_Per-Frame_PCK_{pck_col[-4:]}_vs_Brightness',
            save_dir=config.SAVE_FOLDER
        )

        # Generate the trends plot
        save_filename_trend = f"per_frame_pck_vs_brightness_trends_{pck_col[-4:]}_{timestamp}.png"
        save_path_trend = os.path.join(config.SAVE_FOLDER, save_filename_trend)
        plot_pck_vs_brightness_trends(
            combined_df,
            x_column='brightness',
            y_column=pck_col,
            subject_col=config.SUBJECT_COLUMN,
            action_col=config.ACTION_COLUMN,
            camera_col=config.CAMERA_COLUMN,
            title=f'Per-Frame PCK Score ({pck_col[-4:]}) vs. Video Brightness (Trends)',
            x_label='Per-Frame Video Brightness (L Channel)',
            save_path=save_path_trend
        )

        # Generate the traditional scatter plot
        save_filename_metric = f"per_frame_pck_vs_brightness_{pck_col[-4:]}_{timestamp}.png"
        save_path_metric = os.path.join(
            config.SAVE_FOLDER, save_filename_metric)
        plot_pck_vs_metric(
            df=combined_df,
            x_column='brightness',
            y_column=pck_col,
            subject_col=config.SUBJECT_COLUMN,
            action_col=config.ACTION_COLUMN,
            camera_col=config.CAMERA_COLUMN,
            title=f'PCK ({pck_col[-4:]}) vs Brightness',
            x_label='Brightness (L*)',
            save_path=save_path_metric
        )

    print(f"\nAnalysis complete. Results saved to {config.SAVE_FOLDER}")

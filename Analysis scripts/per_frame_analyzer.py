# per_frame_analyzer.py

import os
import pandas as pd
from datetime import datetime
from HumanEva_config import (
    VIDEO_DIRECTORY,
    PCK_FILE_PATH,
    PCK_PER_FRAME_SCORE_COLUMNS,
    SAVE_FOLDER,
    SUBJECT_COLUMN,
    ACTION_COLUMN,
    CAMERA_COLUMN
)
from video_analyzer import VideoAnalyzer
from pck_data_processor import PCKDataProcessor
from visualizer import (
    plot_pck_vs_metric,
    plot_pck_vs_brightness_trends,
    plot_pck_vs_brightness_interactive)
from video_file_mapper import find_video_for_pck_row


def run_per_frame_analysis():
    """
    Performs per-frame analysis and generates plots with frame number labels.
    Now includes enhanced PCK vs brightness plot with frame number visualization.
    """
    print("\nRunning enhanced per-frame analysis...")
    all_per_frame_data = []

    # Load PCK data
    pck_per_frame_processor = PCKDataProcessor(
        PCK_FILE_PATH, PCK_PER_FRAME_SCORE_COLUMNS)
    pck_per_frame_df = pck_per_frame_processor.load_pck_per_frame_scores()

    if pck_per_frame_df is None:
        print("Cannot proceed with per-frame analysis without data.")
        return

    # Process each video
    grouped_pck_df = pck_per_frame_df.groupby(
        [SUBJECT_COLUMN, ACTION_COLUMN, CAMERA_COLUMN])

    for (subject, action, camera), video_pck_df in grouped_pck_df:
        video_row = pd.Series({
            SUBJECT_COLUMN: subject,
            ACTION_COLUMN: action,
            CAMERA_COLUMN: camera
        })

        # Find matching video file
        video_path = find_video_for_pck_row(VIDEO_DIRECTORY, video_row)
        if not (video_path and os.path.exists(video_path)):
            print(
                f"Warning: Video not found for S{subject}, {action}, C{camera}. Skipping.")
            continue

        # Get brightness data
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
        video_pck_df_aligned['video_id'] = f"S{subject}_{action}_C{camera}"
        video_pck_df_aligned['frame_idx'] = range(min_length)
        video_pck_df_aligned['subject'] = subject  # Add subject column
        video_pck_df_aligned['action'] = action    # Add action column
        video_pck_df_aligned['camera'] = camera    # Add camera column

        all_per_frame_data.append(video_pck_df_aligned)

    if not all_per_frame_data:
        print("No video data could be successfully processed and matched with PCK scores.")
        return

    combined_df = pd.concat(all_per_frame_data, ignore_index=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Generate plots for each PCK threshold
    for pck_col in PCK_PER_FRAME_SCORE_COLUMNS:
        # 4. New: PCK vs Brightness with frame index annotations
        plot_pck_vs_brightness_interactive(
            combined_df,
            x_column='brightness',
            y_column=pck_col,
            subject_col=SUBJECT_COLUMN,
            action_col=ACTION_COLUMN,
            camera_col=CAMERA_COLUMN,
            frame_col='frame_idx',
            title=f'Interactive_Per-Frame_PCK_{pck_col[-4:]}_vs_Brightness',
            save_dir=SAVE_FOLDER
        )

    #     # 2. Enhanced PCK vs brightness plot with frame numbers
    #     save_filename_trend = f"per_frame_pck_vs_brightness_trends_{pck_col[-4:]}_{timestamp}.png"
    #     save_path_trend = os.path.join(SAVE_FOLDER, save_filename_trend)
    #     plot_pck_vs_brightness_trends(
    #         combined_df,
    #         x_column='brightness',
    #         y_column=pck_col,
    #         subject_col=SUBJECT_COLUMN,
    #         action_col=ACTION_COLUMN,
    #         camera_col=CAMERA_COLUMN,
    #         title=f'Per-Frame PCK Score ({pck_col[-4:]}) vs. Video Brightness (Trends)',
    #         x_label='Per-Frame Video Brightness (L Channel)',
    #         save_path=save_path_trend
    #     )

    #     # 3. Optional: Traditional scatter plot (keep if needed)
    #     save_filename_metric = f"pcer_frame_pck_vs_brightness_{pck_col[-4:]}_{timestamp}.png"
    #     save_path_metric = os.path.join(SAVE_FOLDER, save_filename_metric)
    #     plot_pck_vs_metric(
    #         df=combined_df,
    #         x_column='brightness',
    #         y_column=pck_col,
    #         subject_col='subject',
    #         action_col='action',
    #         camera_col='camera',
    #         title=f'PCK ({pck_col[-4:]}) vs Brightness',
    #         x_label='Brightness (L*)',
    #         save_path=save_path_metric
    #     )

    # print(f"\nAnalysis complete. Results saved to {SAVE_FOLDER}")

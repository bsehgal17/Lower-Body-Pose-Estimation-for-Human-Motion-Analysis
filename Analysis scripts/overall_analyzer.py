# overall_analyzer.py

import os
import pandas as pd
import numpy as np
from datetime import datetime
from HumanEva_config import (
    VIDEO_DIRECTORY,
    PCK_FILE_PATH,
    PCK_OVERALL_SCORE_COLUMNS,
    SAVE_FOLDER,
    SUBJECT_COLUMN,
    ACTION_COLUMN,
    CAMERA_COLUMN
)
from video_analyzer import VideoAnalyzer
from pck_data_processor import PCKDataProcessor
from visualizer import (
    plot_brightness_distribution,
    plot_overall_relation,
    plot_pck_vs_metric
)
from video_file_mapper import find_video_for_pck_row


def run_overall_analysis():
    """Performs overall video analysis and generates scatter plots."""
    print("Running overall analysis...")

    # Step 1: Collect ALL brightness and contrast data
    all_brightness_data = []
    all_contrast_data = []
    video_files = []
    for root, _, files in os.walk(VIDEO_DIRECTORY):
        for f in files:
            if f.endswith(('.mp4', '.avi', '.mov')):
                video_files.append(os.path.join(root, f))

    if not video_files:
        print("No video files found for overall analysis. Exiting.")
        return

    print(f"Found {len(video_files)} videos for overall analysis.")
    for video_file in video_files:
        analyzer = VideoAnalyzer(video_file)
        brightness_per_frame = analyzer.get_brightness_data()
        contrast_per_frame = analyzer.get_contrast_data()
        if brightness_per_frame:
            all_brightness_data.extend(brightness_per_frame)
        if contrast_per_frame:
            all_contrast_data.extend(contrast_per_frame)

    if not all_brightness_data:
        print("Could not process any videos. Exiting.")
        return

    overall_avg_l = np.mean(all_brightness_data)
    overall_avg_contrast = np.mean(all_contrast_data)
    print(
        f"\nOverall average brightness across all videos: {overall_avg_l:.2f}")
    print(
        f"Overall average contrast across all videos: {overall_avg_contrast:.2f}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    brightness_plot_path = os.path.join(
        SAVE_FOLDER, f"brightness_distribution_{timestamp}.png")
    plot_brightness_distribution(
        all_brightness_data, save_path=brightness_plot_path)

    # Step 2: Load PCK scores and calculate overall averages
    pck_processor = PCKDataProcessor(PCK_FILE_PATH, PCK_OVERALL_SCORE_COLUMNS)
    pck_df = pck_processor.load_pck_scores()

    if pck_df is None:
        print("Cannot proceed without PCK score data.")
        return

    overall_avg_pck_scores = {
        col: pck_df[col].mean() for col in PCK_OVERALL_SCORE_COLUMNS}
    print("\nOverall Average PCK Scores for the dataset:")
    for col, avg_val in overall_avg_pck_scores.items():
        print(f"- {col}: {avg_val:.4f}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    relation_plot_path = os.path.join(
        SAVE_FOLDER, f"overall_relation_{timestamp}.png")
    plot_overall_relation(
        overall_avg_l, overall_avg_pck_scores, save_path=relation_plot_path)

    # Step 3: Perform per-video analysis and create scatter plots
    pck_df['avg_brightness'] = None
    pck_df['avg_contrast'] = None
    print("\nProcessing videos and linking metrics to PCK scores...")
    for index, row in pck_df.iterrows():
        video_path = find_video_for_pck_row(VIDEO_DIRECTORY, row)

        if video_path and os.path.exists(video_path):
            analyzer = VideoAnalyzer(video_path)
            brightness_data = analyzer.get_brightness_data()
            contrast_data = analyzer.get_contrast_data()
            if brightness_data and contrast_data:
                avg_brightness = np.mean(brightness_data)
                avg_contrast = np.mean(contrast_data)
                pck_df.loc[index, 'avg_brightness'] = avg_brightness
                pck_df.loc[index, 'avg_contrast'] = avg_contrast
                print(
                    f"Processed: {video_path} -> Avg. Brightness: {avg_brightness:.2f}, Avg. Contrast: {avg_contrast:.2f}")
            else:
                print(f"Warning: Could not get metrics for {video_path}")
        else:
            print(
                f"Warning: Video not found for Subject: {row['subject']}, Action: {row['action']}, Camera: {row['camera']}")

    pck_df = pck_df.dropna(subset=['avg_brightness', 'avg_contrast'])
    pck_df['avg_brightness'] = pd.to_numeric(pck_df['avg_brightness'])
    pck_df['avg_contrast'] = pd.to_numeric(pck_df['avg_contrast'])

    for pck_col in PCK_OVERALL_SCORE_COLUMNS:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_filename = f"pck_vs_brightness_{pck_col[-4:]}_{timestamp}.png"
        save_path = os.path.join(SAVE_FOLDER, save_filename)
        plot_pck_vs_metric(
            pck_df,
            x_column='avg_brightness',
            y_column=pck_col,
            subject_col=SUBJECT_COLUMN,
            action_col=ACTION_COLUMN,
            camera_col=CAMERA_COLUMN,
            title=f'PCK Score ({pck_col[-4:]}) vs. Average Video Brightness',
            x_label='Average Video Brightness (L Channel)',
            save_path=save_path
        )

    for pck_col in PCK_OVERALL_SCORE_COLUMNS:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_filename = f"pck_vs_contrast_{pck_col[-4:]}_{timestamp}.png"
        save_path = os.path.join(SAVE_FOLDER, save_filename)
        plot_pck_vs_metric(
            pck_df,
            x_column='avg_contrast',
            y_column=pck_col,
            subject_col=SUBJECT_COLUMN,
            action_col=ACTION_COLUMN,
            camera_col=CAMERA_COLUMN,
            title=f'PCK Score ({pck_col[-4:]}) vs. Average Video Contrast',
            x_label='Average Video Contrast (L Channel Standard Deviation)',
            save_path=save_path
        )

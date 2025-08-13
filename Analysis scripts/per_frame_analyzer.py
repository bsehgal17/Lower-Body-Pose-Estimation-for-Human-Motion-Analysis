# per_frame_analyzer.py

import os
import pandas as pd
from datetime import datetime
from HumanEva_config import (
    VIDEO_DIRECTORY,
    PCK_FILE_PATH,
    PCK_SCORE_COLUMNS,
    SAVE_FOLDER
)
from video_analyzer import VideoAnalyzer
from pck_data_processor import PCKDataProcessor
from visualizer import plot_per_frame_timeseries
from video_file_mapper import find_video_for_pck_row


def run_per_frame_analysis():
    """Performs per-frame analysis and generates time series plots."""
    print("\nRunning per-frame analysis...")
    all_per_frame_data = []

    pck_per_frame_processor = PCKDataProcessor(
        PCK_FILE_PATH, PCK_SCORE_COLUMNS)
    pck_per_frame_df = pck_per_frame_processor.load_pck_per_frame_scores()

    if pck_per_frame_df is None:
        print("Cannot proceed with per-frame analysis without data.")
        return

    # Merge per-frame PCK with per-frame video metrics
    for _, row in pck_per_frame_df.iterrows():
        video_path = find_video_for_pck_row(VIDEO_DIRECTORY, row)
        if video_path and os.path.exists(video_path):
            analyzer = VideoAnalyzer(video_path)
            brightness_data = analyzer.get_brightness_data()
            if brightness_data:
                # Assuming number of frames matches the PCK data
                per_frame_df = pd.DataFrame({
                    'frame_idx': range(len(brightness_data)),
                    'brightness': brightness_data,
                    # Assuming this is a list or series
                    'pck_score': row['pck_score'],
                    'video_id': f"{row['subject']}_{row['action']}_{row['camera']}"
                })
                all_per_frame_data.append(per_frame_df)

    if all_per_frame_data:
        combined_df = pd.concat(all_per_frame_data, ignore_index=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(
            SAVE_FOLDER, f"per_frame_pck_and_brightness_{timestamp}.png")
        plot_per_frame_timeseries(
            combined_df,
            pck_score_columns=PCK_SCORE_COLUMNS,
            save_path=save_path
        )

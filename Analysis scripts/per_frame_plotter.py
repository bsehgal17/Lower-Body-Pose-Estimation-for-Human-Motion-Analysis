# per_frame_analyzer.py

import os
from datetime import datetime
import pandas as pd
from visualizer import (
    plot_pck_vs_metric,
    plot_pck_vs_brightness_trends,
    plot_pck_vs_brightness_interactive
)


def plot_per_frame_analysis(config, combined_df, metric_name):
    """
    Generates all per-frame plots for a specified metric.

    Args:
        config (object): A configuration object containing dataset-specific parameters.
        combined_df (pd.DataFrame): The DataFrame with combined PCK and metric data.
        metric_name (str): The name of the metric to plot (e.g., 'brightness', 'contrast').
    """
    print("\n" + "=" * 50)
    print(f"Generating per-frame plots for {metric_name}...")

    if combined_df.empty or metric_name not in combined_df.columns:
        print(f"No {metric_name} data to plot. Skipping analysis.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # --- Data Cleaning Step to handle the error ---
    # The error "Cannot convert non-finite values (NA or inf) to integer"
    # suggests missing data. This step cleans the DataFrame before plotting.
    if combined_df.isnull().values.any():
        print(f"Warning: The DataFrame contains missing values. Handling them...")
        # Fill missing values in the metric and PCK columns with 0
        columns_to_fill = [metric_name] + config.PCK_PER_FRAME_SCORE_COLUMNS
        combined_df.loc[:, columns_to_fill] = combined_df.loc[:,
                                                              columns_to_fill].fillna(0)
        # Drop any remaining rows with NaN values that could cause issues
        combined_df.dropna(inplace=True)

    # Generate plots for all videos together
    print("\nGenerating combined plots for all videos...")
    for pck_col in config.PCK_PER_FRAME_SCORE_COLUMNS:
        plot_pck_vs_brightness_interactive(
            combined_df,
            x_column=metric_name,
            y_column=pck_col,
            subject_col=config.SUBJECT_COLUMN,
            action_col=config.ACTION_COLUMN,
            camera_col=config.CAMERA_COLUMN,
            frame_col='frame_idx',
            title=f'Interactive_Per-Frame_LOWER_PCK_{pck_col[-4:]}_vs_{metric_name.title()} ({config.DATASET_NAME} {config.MODEL})',
            save_dir=config.SAVE_FOLDER
        )

        save_filename_trend = f"per_frame_pck_vs_{metric_name}_trends_{pck_col[-4:]}_{timestamp}.png"
        save_path_trend = os.path.join(config.SAVE_FOLDER, save_filename_trend)
        plot_pck_vs_brightness_trends(
            combined_df,
            x_column=metric_name,
            y_column=pck_col,
            subject_col=config.SUBJECT_COLUMN,
            action_col=config.ACTION_COLUMN,
            camera_col=config.CAMERA_COLUMN,
            title=f'Per-Frame LOWER PCK Score ({pck_col[-4:]}) vs. Video {metric_name.title()} (Trends) ({config.DATASET_NAME} {config.MODEL})',
            x_label=f'Per-Frame Video {metric_name.title()}',
            save_path=save_path_trend
        )

        save_filename_metric = f"per_frame_pck_vs_{metric_name}_{pck_col[-4:]}_{timestamp}.png"
        save_path_metric = os.path.join(
            config.SAVE_FOLDER, save_filename_metric)
        plot_pck_vs_metric(
            df=combined_df,
            x_column=metric_name,
            y_column=pck_col,
            subject_col=config.SUBJECT_COLUMN,
            action_col=config.ACTION_COLUMN,
            camera_col=config.CAMERA_COLUMN,
            title=f'LOWER PCK ({pck_col[-4:]}) vs {metric_name.title()} ({config.DATASET_NAME} {config.MODEL})',
            x_label=f'{metric_name.title()} (L*)',
            save_path=save_path_metric
        )

    # =========================================================================
    # New functionality: Generate plots for each video individually
    # =========================================================================
    print("\n" + "=" * 50)
    print("Generating per-frame plots for each individual video...")

    # Create the subfolder for video-specific plots
    video_save_dir = os.path.join(config.SAVE_FOLDER, 'videos')
    os.makedirs(video_save_dir, exist_ok=True)

    # Dynamically build the list of grouping columns, ignoring None values
    grouping_cols = [col for col in [config.SUBJECT_COLUMN,
                                     config.ACTION_COLUMN, config.CAMERA_COLUMN] if col is not None]

    if not grouping_cols:
        print("Warning: No valid grouping columns found. Cannot generate per-video plots.")
        return

    unique_videos = combined_df.groupby(grouping_cols)

    for name, group in unique_videos:
        # The 'name' from groupby is a tuple when grouping by multiple columns,
        # but a single value when grouping by one column. This handles both cases.
        video_name_parts = name if isinstance(name, tuple) else (name,)

        # Create a unique video ID string for titles and filenames
        video_id_parts = [f"{col}{video_name_parts[i]}" for i,
                          col in enumerate(grouping_cols)]
        video_id_label = "_".join(video_id_parts)

        title_prefix = f"Video {video_id_label}:"
        save_file_prefix = f"video_{video_id_label}"

        for pck_col in config.PCK_PER_FRAME_SCORE_COLUMNS:
            # Generate and save the scatter plot for this single video
            save_filename_metric = f"{save_file_prefix}_pck_vs_{metric_name}_{pck_col[-4:]}_{timestamp}.png"
            save_path_metric = os.path.join(
                video_save_dir, save_filename_metric)
            plot_pck_vs_metric(
                df=group,  # Pass the filtered DataFrame for a single video
                x_column=metric_name,
                y_column=pck_col,
                subject_col=None,  # Pass None since we're plotting a single video
                action_col=None,
                camera_col=None,
                title=f'{title_prefix} LOWER PCK ({pck_col[-4:]}) vs {metric_name.title()}',
                x_label=f'{metric_name.title()} (L*)',
                save_path=save_path_metric
            )

            # Generate and save the trends plot for this single video
            save_filename_trend = f"{save_file_prefix}_pck_vs_{metric_name}_trends_{pck_col[-4:]}_{timestamp}.png"
            save_path_trend = os.path.join(video_save_dir, save_filename_trend)
            plot_pck_vs_brightness_trends(
                df=group,  # Pass the filtered DataFrame
                x_column=metric_name,
                y_column=pck_col,
                subject_col=None,  # Pass None
                action_col=None,
                camera_col=None,
                title=f'{title_prefix} Per-Frame LOWER PCK Score ({pck_col[-4:]}) vs. Video {metric_name.title()} (Trends)',
                x_label=f'Per-Frame Video {metric_name.title()}',
                save_path=save_path_trend
            )

    print("\nPer-frame analysis complete.")

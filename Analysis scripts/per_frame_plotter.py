import os
from datetime import datetime
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

    for pck_col in config.PCK_PER_FRAME_SCORE_COLUMNS:
        plot_pck_vs_brightness_interactive(
            combined_df,
            x_column=metric_name,
            y_column=pck_col,
            subject_col=config.SUBJECT_COLUMN,
            action_col=config.ACTION_COLUMN,
            camera_col=config.CAMERA_COLUMN,
            frame_col='frame_idx',
            title=f'Interactive_Per-Frame_LOWER_PCK_{pck_col[-4:]}_vs_{metric_name.title()}',
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
            title=f'Per-Frame LOWER PCK Score ({pck_col[-4:]}) vs. Video {metric_name.title()} (Trends)',
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
            title=f'LOWER PCK ({pck_col[-4:]}) vs {metric_name.title()}',
            x_label=f'{metric_name.title()} (L*)',
            save_path=save_path_metric
        )

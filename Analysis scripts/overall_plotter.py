import os
from datetime import datetime
from visualizer import (
    plot_overall_relation,
    plot_pck_vs_metric,
    plot_overall_distribution
)


def plot_overall_analysis(config, merged_df, all_metric_data, metric_name):
    """
    Generates all overall plots for a given video metric.

    Args:
        config (object): A configuration object.
        merged_df (pd.DataFrame): DataFrame with combined PCK and average metric scores.
        all_metric_data (list): A list of all per-frame data points for the metric.
        metric_name (str): The name of the metric being analyzed (e.g., 'brightness').
    """
    print("\n" + "=" * 50)
    print(f"Generating overall plots for {metric_name}...")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    avg_metric_column = f'avg_{metric_name}'

    # Check if the required data exists before plotting
    if merged_df.empty or avg_metric_column not in merged_df.columns:
        print(f"No combined data for {metric_name}. Skipping plots.")
        return

    # Plot 1: Overall distribution of the metric
    plot_overall_distribution(
        all_metric_data,
        metric_name=metric_name.title(),
        units='L* Channel, 0-255',
        save_path=os.path.join(
            config.SAVE_FOLDER, f'overall_{metric_name}_distribution_{timestamp}.png')
    )

    # Plot 2: Overall relation between average metric and average PCK
    overall_avg_metric = merged_df[avg_metric_column].mean()
    avg_pck_scores = {col: merged_df[col].mean()
                      for col in config.PCK_OVERALL_SCORE_COLUMNS}
    plot_overall_relation(
        overall_avg_metric,
        avg_pck_scores,
        save_path=os.path.join(
            config.SAVE_FOLDER, f'overall_pck_vs_{metric_name}_{timestamp}.png')
    )

    # Plot 3: Individual PCK score vs average metric scatter plot
    for pck_col in config.PCK_OVERALL_SCORE_COLUMNS:
        plot_pck_vs_metric(
            df=merged_df,
            x_column=avg_metric_column,
            y_column=pck_col,
            subject_col=config.SUBJECT_COLUMN,
            action_col=config.ACTION_COLUMN,
            camera_col=config.CAMERA_COLUMN,
            title=f'LOWER PCK ({pck_col[-4:]}) vs Average {metric_name.title()}',
            x_label=f'Average Video {metric_name.title()} (L*)',
            save_path=os.path.join(
                config.SAVE_FOLDER, f'overall_pck_vs_avg_{metric_name}_{pck_col[-4:]}_{timestamp}.png')
        )

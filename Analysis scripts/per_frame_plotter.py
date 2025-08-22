import os
from datetime import datetime
import pandas as pd
import numpy as np
from visualizer import (
    plot_pck_vs_metric,
    plot_pck_vs_metric_combined  # Importing the new function
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
    if combined_df.isnull().values.any():
        print(f"Warning: The DataFrame contains missing values. Handling them...")
        columns_to_fill = [metric_name] + config.PCK_PER_FRAME_SCORE_COLUMNS
        combined_df.loc[:, columns_to_fill] = combined_df.loc[:,
                                                              columns_to_fill].fillna(0)
        combined_df.dropna(inplace=True)

    # Generate the new plots with a single color and no legend
    print("\nGenerating new combined plots with a single color...")
    for pck_col in config.PCK_PER_FRAME_SCORE_COLUMNS:
        new_save_filename_metric = f"per_frame_pck_vs_{metric_name}_{pck_col[-4:]}_combined_{timestamp}.svg"
        new_save_path_metric = os.path.join(
            config.SAVE_FOLDER, new_save_filename_metric)
        plot_pck_vs_metric_combined(
            df=combined_df,
            x_column=metric_name,
            y_column=pck_col,
            title=f'LOWER PCK ({pck_col[-4:]}) vs {metric_name.title()} ({config.DATASET_NAME} {config.MODEL})',
            x_label=f'{metric_name.title()} (L*)',
            save_path=new_save_path_metric
        )

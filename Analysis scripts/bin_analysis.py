import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def analyze_frames_and_pck(config, df, metric_name):
    """
    Bins a metric, plots number of frames per bin, computes PCK statistics,
    and saves/updates a summary Excel file for each metric (no timestamp).

    Args:
        config (object): Config object with SAVE_FOLDER and PCK_PER_FRAME_SCORE_COLUMNS.
        df (pd.DataFrame): DataFrame with per-frame data.
        metric_name (str): Metric column to analyze (e.g., 'brightness').
    """
    print("\n" + "="*50)
    print(f"Analyzing Metric: {metric_name.title()}")

    # --- Data Cleaning ---
    if df.isnull().values.any():
        print("Warning: Missing values found. Filling with 0...")
        columns_to_fill = [metric_name] + \
            getattr(config, 'PCK_PER_FRAME_SCORE_COLUMNS', [])
        df.loc[:, columns_to_fill] = df.loc[:, columns_to_fill].fillna(0)
        df.dropna(inplace=True)

    # --- Binning ---
    if metric_name == 'brightness':
        bins = [0, 50, 100, 150, 200, 255]
        labels = ['Low', 'Medium-Low', 'Medium', 'Medium-High', 'High']
    else:
        bins = pd.qcut(df[metric_name], q=5, retbins=True, labels=False)[1]
        labels = [f'Bin {i+1}' for i in range(5)]

    df[f'{metric_name}_bin'] = pd.cut(
        df[metric_name], bins=bins, labels=labels, right=False)

    # --- Number of frames per bin ---
    frame_counts = df[f'{metric_name}_bin'].value_counts().reindex(
        labels, fill_value=0)
    print("\nNumber of frames per bin:")
    print(frame_counts)

    # --- Plot number of frames per bin ---
    plt.figure(figsize=(10, 6))
    sns.barplot(x=frame_counts.index, y=frame_counts.values, palette='viridis')
    plt.title(f'Number of Frames per {metric_name.title()} Bin', fontsize=16)
    plt.xlabel(f'{metric_name.title()} Bin', fontsize=12)
    plt.ylabel('Number of Frames', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    if not os.path.exists(config.SAVE_FOLDER):
        os.makedirs(config.SAVE_FOLDER)

    plot_path = os.path.join(
        config.SAVE_FOLDER, f'{metric_name}_frame_counts.svg')
    plt.savefig(plot_path, bbox_inches='tight')
    print(f"Plot saved to '{plot_path}'")
    # plt.show()

    # --- Compute PCK statistics per bin ---
    pck_summary_list = []
    for pck_col in config.PCK_PER_FRAME_SCORE_COLUMNS:
        stats = df.groupby(f'{metric_name}_bin')[pck_col].agg(
            mean_pck='mean',
            median_pck='median',
            std_pck='std',
            count='count'
        ).reset_index()
        stats['PCK_Column'] = pck_col
        pck_summary_list.append(stats)

    summary_df = pd.concat(pck_summary_list, ignore_index=True)

    # --- Save/update Excel file per metric ---
    summary_file = os.path.join(
        config.SAVE_FOLDER, f'{metric_name}_summary.xlsx')
    if os.path.exists(summary_file):
        existing_df = pd.read_excel(summary_file)
        combined_df = pd.concat([existing_df, summary_df], ignore_index=True)
        combined_df.to_excel(summary_file, index=False)
        print(f"Updated summary for {metric_name} saved to '{summary_file}'")
    else:
        summary_df.to_excel(summary_file, index=False)
        print(f"Summary for {metric_name} created at '{summary_file}'")

    print("="*50 + "\nAnalysis Completed.")

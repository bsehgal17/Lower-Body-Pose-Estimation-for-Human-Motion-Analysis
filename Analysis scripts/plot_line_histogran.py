import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_all_thresholds_pck_distribution(config, df):
    """
    Plots line distributions of PCK scores (40–120) for all thresholds in one plot.

    Args:
        config (object): Config object with SAVE_FOLDER and PCK_PER_FRAME_SCORE_COLUMNS.
        df (pd.DataFrame): DataFrame containing per-frame PCK scores for multiple thresholds.
    """
    print("\n" + "="*50)
    print("Running PCK Multi-Threshold Line Plot...")

    if not hasattr(config, "PCK_PER_FRAME_SCORE_COLUMNS"):
        raise ValueError(
            "Config must have PCK_PER_FRAME_SCORE_COLUMNS attribute.")

    pck_cols = [
        col for col in config.PCK_PER_FRAME_SCORE_COLUMNS if col in df.columns]
    if not pck_cols:
        print("⚠️ No valid PCK columns found in DataFrame.")
        return

    # Define bins (integers from 40 to 120)
    bins = list(range(-2, 102))

    plt.figure(figsize=(12, 7))

    for pck_col in pck_cols:
        # Round values to nearest int and count
        counts = df[pck_col].round().astype(
            int).value_counts().reindex(bins, fill_value=0)
        plt.plot(counts.index, counts.values, marker="o",
                 linestyle="-", label=pck_col)

    # Labels and formatting
    plt.title("Frame Count per PCK Score (All Thresholds)")
    plt.xlabel("PCK Score")
    plt.ylabel("Number of Frames")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(title="Thresholds", bbox_to_anchor=(1.05, 1), loc="upper left")

    # Save
    save_path = os.path.join(config.SAVE_FOLDER, "pck_all_thresholds.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ Multi-threshold PCK line plot saved to {save_path}")
    print("="*50 + "\nPlot Completed.")

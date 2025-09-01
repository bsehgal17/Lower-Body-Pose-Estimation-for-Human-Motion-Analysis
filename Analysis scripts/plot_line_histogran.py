import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_brightness_overlay_and_stats(config, df, bin_width=2, normalize=True):
    """
    Plots brightness distributions for PCK=0 and PCK=100 (line histogram)
    and calculates mean, std, IQR for all PCK groups.

    Args:
        config (object): Config object with SAVE_FOLDER and PCK_PER_FRAME_SCORE_COLUMNS.
        df (pd.DataFrame): DataFrame with columns ["brightness"] and multiple PCK columns.
        bin_width (int): Width of brightness bins (default=2).
        normalize (bool): Normalize y-axis by total frames.
    """
    print("\n" + "=" * 60)
    print("Running Brightness Overlay + Statistics by PCK Scores...")

    if not hasattr(config, "PCK_PER_FRAME_SCORE_COLUMNS"):
        raise ValueError(
            "Config must have PCK_PER_FRAME_SCORE_COLUMNS attribute.")

    valid_pck_cols = [
        col for col in config.PCK_PER_FRAME_SCORE_COLUMNS if col in df.columns]
    if not valid_pck_cols:
        print("⚠️ No valid PCK columns found in DataFrame.")
        return

    # ------------------- Plotting -------------------
    bins = np.arange(df["brightness"].min(),
                     df["brightness"].max() + bin_width, bin_width)
    plt.figure(figsize=(12, 7))

    for pck_col in valid_pck_cols:
        subset = df[["brightness", pck_col]].dropna()
        if subset.empty:
            continue

        subset["PCK_Group"] = subset[pck_col].round().astype(int)

        for score in sorted(subset["PCK_Group"].unique()):
            if score != 0 or score != 100:
                continue
            brightness_values = subset.loc[subset["PCK_Group"]
                                           == score, "brightness"].values
            if len(brightness_values) == 0:
                continue

            counts, bin_edges = np.histogram(brightness_values, bins=bins)
            if normalize:
                counts = counts / counts.sum()

            plt.plot(
                bin_edges[:-1],
                counts,
                linestyle="-",
                marker="o",
                markersize=3,
                label=f"{pck_col} | PCK={score}"
            )

    plt.title("Brightness Distribution Overlaid ")
    plt.xlabel("Brightness")
    plt.ylabel("Relative Frequency" if normalize else "Frame Count")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    save_path = os.path.join(
        config.SAVE_FOLDER, f"brightness_pck_overlay{pck_col}(0 and 100).svg")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Overlay plot saved to {save_path}")

    # ------------------- Statistics -------------------
    stats_list = []
    for pck_col in valid_pck_cols:
        subset = df[["brightness", pck_col]].dropna()
        subset["PCK_Group"] = subset[pck_col].round().astype(int)

        for score in sorted(subset["PCK_Group"].unique()):
            brightness_values = subset.loc[subset["PCK_Group"]
                                           == score, "brightness"].values
            if len(brightness_values) == 0:
                continue

            mean_val = np.mean(brightness_values)
            std_val = np.std(brightness_values)
            iqr_val = np.percentile(brightness_values, 75) - \
                np.percentile(brightness_values, 25)

            stats_list.append({
                "PCK_Column": pck_col,
                "PCK_Group": score,
                "Mean_Brightness": mean_val,
                "Std_Deviation": std_val,
                "IQR": iqr_val,
                "Frame_Count": len(brightness_values)
            })

    stats_df = pd.DataFrame(stats_list)

    # Save/update Excel
    stats_file = os.path.join(
        config.SAVE_FOLDER, f"brightness_pck_statistics{pck_col}.xlsx")
    if os.path.exists(stats_file):
        existing_df = pd.read_excel(stats_file)
        combined_df = pd.concat([existing_df, stats_df], ignore_index=True)
        combined_df.to_excel(stats_file, index=False)
        print(f"Updated statistics saved to '{stats_file}'")
    else:
        stats_df.to_excel(stats_file, index=False)
        print(f"Statistics created at '{stats_file}'")

    print("=" * 60 + "\nOverlay and Statistics Completed.")

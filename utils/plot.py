import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
from typing import Dict, Optional, Tuple


def plot_filtering_effect(
    original, filtered, title="Signal Filtering Comparison", save_path=None
):
    """
    Plots original vs filtered signals on the same plot.

    Args:
        original (array): Raw signal data.
        filtered (array): Processed signal data.
        title (str): Plot title.
        save_path (str): If provided, saves plot to this path.
    """
    plt.figure()
    if len(original) == 1:
        plt.plot(original, label="Original", linestyle="None", marker="o")
    else:
        plt.plot(original, label="Original", linestyle="--")

    if len(filtered) == 1:
        plt.plot(filtered, label="Filtered", linestyle="None", marker="x")
    else:
        plt.plot(filtered, label="Filtered", linewidth=2)

    plt.title(title)
    plt.xlabel("Frame Number")
    plt.ylabel("Coordinate Value")
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_metric_boxplot_from_files(
    label_file_map: Dict[str, str],
    metric_column: Optional[str] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot a metric boxplot given a mapping of labels to Excel files.

    Args:
        label_file_map (Dict[str, str]): Mapping from label (e.g., 'x-20') to Excel file path.
        metric_column (Optional[str]): Column to extract; prompts if None.
        title (Optional[str]): Title for the plot.
        save_path (Optional[str]): If provided, saves the plot to this path.
    """
    long_data = []

    for label, file_path in label_file_map.items():
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        df = pd.read_excel(file_path)

        # Detect column if not provided
        if metric_column is None:
            print(f"[{label}] Available columns: {list(df.columns)}")
            metric_column = input("Enter the metric column name to plot: ").strip()

        if metric_column not in df.columns:
            raise ValueError(f"Column '{metric_column}' not found in {file_path}")

        for val in df[metric_column].dropna():
            long_data.append({"Label": label, "Metric": val})

    long_df = pd.DataFrame(long_data)

    # Plotting
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=long_df,
        x="Label",
        y="Metric",
        showmeans=True,
        meanprops={
            "marker": "o",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
            "markersize": 7,
        },
    )
    plt.title(title or f"{metric_column} Comparison")
    plt.grid(True, axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    else:
        plt.show()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.cm as cm


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_brightness_distribution(all_brightness_data, save_path=None):
    """
    Plots a histogram of the brightness distribution across all frames in the dataset
    with enhanced features and a fixed x-axis range.
    """
    if not all_brightness_data:
        print("No brightness data to plot.")
        return

    # Use a better plotting style
    sns.set_style("whitegrid")

    # Calculate statistics for annotations
    mean_brightness = np.mean(all_brightness_data)
    std_dev_brightness = np.std(all_brightness_data)
    num_frames = len(all_brightness_data)

    plt.figure(figsize=(10, 6))

    # Create the histogram
    plt.hist(all_brightness_data, bins=30,
             color='lightcoral', edgecolor='black', alpha=0.8)

    # Add a vertical line for the mean
    plt.axvline(mean_brightness, color='darkred', linestyle='dashed',
                linewidth=2, label=f'Mean: {mean_brightness:.2f}')

    # Add text annotations for key statistics
    plt.text(0.95, 0.90, f"Mean: {mean_brightness:.2f}\nStd Dev: {std_dev_brightness:.2f}",
             transform=plt.gca().transAxes,
             fontsize=12,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.6))

    plt.title(
        f'Overall Brightness Distribution ({num_frames} Frames)', fontsize=16)
    plt.xlabel('Mean Frame Brightness (L* Channel, 0-255)', fontsize=12)
    plt.ylabel('Frequency (Number of Frames)', fontsize=12)

    # Set the x-axis to go from 0 to 255
    plt.xlim(0, 255)

    plt.legend()
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Brightness distribution plot saved to {save_path}")

    plt.show()


def plot_overall_relation(overall_avg_l, avg_pck_scores, save_path=None):
    """
    Plots the overall average brightness against the average PCK scores for each threshold.
    """
    pck_thresholds = [col[-4:] for col in avg_pck_scores.keys()]
    pck_values = list(avg_pck_scores.values())

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # --- Plot PCK Scores on the primary Y-axis (left) ---
    ax1.set_xlabel('PCK Threshold')
    ax1.set_ylabel('Average PCK Score', color='teal')
    ax1.tick_params(axis='y', labelcolor='teal')

    bars = ax1.bar(pck_thresholds, pck_values, color=['skyblue', 'salmon', 'lightgreen'],
                   label='Average PCK Score', width=0.6)

    # Add text annotations for each bar
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2f}',
                 ha='center', va='bottom', color='black')

    # --- Plot Overall Brightness on the secondary Y-axis (right) ---
    ax2 = ax1.twinx()  # Create a second y-axis
    ax2.set_ylabel('Overall Average Brightness (L)', color='darkred')
    ax2.tick_params(axis='y', labelcolor='darkred')

    # Add a horizontal line for the brightness on the second axis
    ax2.axhline(y=overall_avg_l, color='darkred', linestyle='--', linewidth=2)

    # Add annotation for the brightness line
    ax2.text(len(pck_thresholds) - 1.25, overall_avg_l, f'{overall_avg_l:.2f}',
             ha='left', va='center', color='darkred', weight='bold',
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    # Set the y-limits for the secondary axis to better show the line
    ax2.set_ylim(0, max(overall_avg_l * 1.2, ax1.get_ylim()[1]))

    # --- Final Plot Aesthetics ---
    fig.suptitle(
        'Overall Average PCK Scores vs. Overall Average Brightness', fontsize=14)
    fig.tight_layout()  # Adjusts plot to prevent labels from overlapping
    fig.legend(loc='upper right', bbox_to_anchor=(
        1, 1), bbox_transform=ax1.transAxes)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    if save_path:
        plt.savefig(save_path)
        print(f"Overall relation plot saved to {save_path}")

    plt.show()


def plot_pck_vs_metric(df, x_column, y_column, subject_col, action_col, camera_col, title, x_label, save_path=None):
    """
    Plots a scatter plot of PCK scores vs. a given metric (brightness, contrast, etc.).

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        x_column (str): The name of the metric column (e.g., 'avg_brightness').
        y_column (str): The name of the PCK score column.
        subject_col (str): The name of the subject column for grouping.
        action_col (str): The name of the action column for grouping.
        camera_col (str): The name of the camera column for grouping.
        title (str): The title of the plot.
        x_label (str): The label for the x-axis.
        save_path (str): The path to save the plot.
    """
    plt.figure(figsize=(12, 8))

    unique_videos = df.groupby([subject_col, action_col, camera_col])
    colors = cm.get_cmap('tab20', len(unique_videos))

    i = 0
    for name, group in unique_videos:
        subject, action, camera = name
        camera_str = f"C{int(camera) + 1}"
        video_name = f"{action}_{camera_str}"
        label = f"Subject: {subject}, Video: {video_name}"

        plt.scatter(group[x_column], group[y_column],
                    alpha=0.7, color=colors(i), label=label)
        i += 1

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(f'PCK Score ({y_column[-4:]})')
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    if save_path:
        plt.savefig(save_path)
        print(f"Scatter plot saved to {save_path}")

    plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import numpy as np


def plot_pck_vs_brightness_from_files(file_paths, sheet_name, column_name, save_path=None, x_labels_list=None):
    if not file_paths:
        print("Error: No file paths provided.")
        return

    data_to_plot = []
    if x_labels_list and len(x_labels_list) != len(file_paths):
        print("Warning: The number of custom labels does not match the number of files. Falling back to automatic labels.")
        x_labels_list = None
    if not x_labels_list:
        x_labels_list = []

    # Read Excel files
    for file_path in file_paths:
        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            if column_name not in df.columns:
                print(
                    f"Warning: Column '{column_name}' not found in {file_path}. Skipping.")
                if not x_labels_list:
                    x_labels_list.append(None)
                continue

            pck_scores = df[column_name].tolist()
            data_to_plot.append(pck_scores)

            if not x_labels_list:
                dir_name = os.path.basename(os.path.dirname(file_path))
                match = re.search(r'_(\d+)', dir_name)
                reduction_level = int(match.group(1)) if match else 0
                x_labels_list.append(f"x-{reduction_level}")

        except Exception as e:
            print(f"An error occurred with {file_path}: {e}")
            if not x_labels_list:
                x_labels_list.append(None)

    if not data_to_plot:
        print("Error: No valid data found for plotting.")
        return

    # ---- Plot ----
    plt.figure(figsize=(14, 10))

    # Pastel colors for boxes
    box_colors = ['#aec7e8', '#ffbb78', '#98df8a', '#ff9896',
                  '#c5b0d5', '#c49c94', '#f7b6d2', '#c7c7c7']

    # Create boxplot
    bplot = plt.boxplot(data_to_plot, patch_artist=True, labels=x_labels_list)

    # Style boxes
    for i, patch in enumerate(bplot['boxes']):
        patch.set_facecolor(box_colors[i % len(box_colors)])
        patch.set_edgecolor('black')
        patch.set_linewidth(1.5)

    # Whiskers and caps
    for whisker in bplot['whiskers']:
        whisker.set_color('black')
        whisker.set_linewidth(1.5)
    for cap in bplot['caps']:
        cap.set_color('black')
        cap.set_linewidth(1.5)

    # Median lines
    for median_line in bplot['medians']:
        median_line.set_color('black')
        median_line.set_linewidth(1.5)

    # Outliers
    for flier in bplot['fliers']:
        flier.set(marker='o', color='black', alpha=0.7)

    # Add mean markers and summary stats
    for i, current_data in enumerate(data_to_plot):
        median = np.median(current_data)
        mean = np.mean(current_data)
        q1 = np.percentile(current_data, 25)
        std_dev = np.std(current_data)

        # Plot mean as red diamond
        plt.plot(i + 1, mean, marker='D', color='red')

        # Text above box
        text_y_pos = q1 - 5  # slightly above Q3
        stats_text = (f"Median: {median:.3f}\nMean: {mean:.3f}\n"
                      f"SD: {std_dev:.3f}\n")
        plt.text(i + 1, text_y_pos, stats_text,
                 ha='center', va='bottom',
                 fontsize=16, color='black',
                 bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

    plt.title(
        'PCK Score (Threshold: 0.5) vs. Brightness Reduction Level (Dataset: HumanEva, Model: RTMW)', fontsize=20)
    plt.xlabel('Brightness Reduction Level', fontsize=20)
    plt.ylabel('PCK Score', fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, format='svg')
        print(f"Plot saved to {save_path}")

    plt.show()


# Example usage
file_paths_list = [
    '/storage/Projects/Gaitly/bsehgal/lower_body_pose_est/pipeline_results/HumanEva/Perturbed_RTMW_X_20/evaluation/2025-08-20_12-17-33/2025-08-20_12-17-33_metrics.xlsx',
    '/storage/Projects/Gaitly/bsehgal/lower_body_pose_est/pipeline_results/HumanEva/Perturbed_RTMW_X_40/evaluation/2025-08-20_12-15-29/2025-08-20_12-15-29_metrics.xlsx',
    '/storage/Projects/Gaitly/bsehgal/lower_body_pose_est/pipeline_results/HumanEva/Perturbed_RTMW_X_60/evaluation/2025-08-20_12-21-27/2025-08-20_12-21-27_metrics.xlsx',
    '/storage/Projects/Gaitly/bsehgal/lower_body_pose_est/pipeline_results/HumanEva/Perturbed_RTMW_X_80/evaluation/2025-08-20_14-21-45/2025-08-20_14-21-45_metrics.xlsx'
]
sheet_name_to_plot = 'Overall Metrics'
column_to_plot = 'overall_overall_pck_0.50'

custom_labels = ['x-20', 'x-40', 'x-60', 'x-80']

plot_pck_vs_brightness_from_files(
    file_paths_list,
    sheet_name=sheet_name_to_plot,
    column_name=column_to_plot,
    save_path='/storage/Projects/Gaitly/bsehgal/lower_body_pose_est/pipeline_results/analysis_results/pck_vs_brightness_from_files.svg',
    x_labels_list=custom_labels
)

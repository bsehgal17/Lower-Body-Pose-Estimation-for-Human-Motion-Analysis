import os
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.cm as cm
import seaborn as sns


def plot_overall_distribution(data, metric_name, units, save_path=None):
    """
    Plots a histogram of the distribution of any given data with enhanced features.

    Args:
        data (list or np.array): The data to plot (e.g., brightness values, sharpness scores).
        metric_name (str): The name of the metric (e.g., 'Brightness', 'Sharpness').
        units (str): The units for the metric (e.g., 'L* Channel, 0-255', 'Laplacian Variance').
        save_path (str, optional): The full path to save the plot.
                                   If None, the plot is not saved.
    """
    if not data:
        print(f"No {metric_name} data to plot.")
        return

    sns.set_style("whitegrid")
    mean_value = np.mean(data)
    std_dev_value = np.std(data)
    num_frames = len(data)

    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=30,
             color='lightcoral', edgecolor='black', alpha=0.8)
    plt.axvline(mean_value, color='darkred', linestyle='dashed',
                linewidth=2, label=f'Mean: {mean_value:.2f}')

    plt.text(0.95, 0.90, f"Mean: {mean_value:.2f}\nStd Dev: {std_dev_value:.2f}",
             transform=plt.gca().transAxes,
             fontsize=12,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.6))

    plt.title(
        f'Overall {metric_name} Distribution ({num_frames} Frames)', fontsize=16)
    plt.xlabel(f'Mean Frame {metric_name} ({units})', fontsize=12)
    plt.ylabel('Frequency (Number of Frames)', fontsize=12)
    # Automatically determine x-axis limits based on the data
    plt.xlim(min(data) * 0.95, max(data) * 1.05)
    plt.legend()
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()

    if save_path:
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"{metric_name} distribution plot saved to {save_path}")
    plt.close()


def plot_overall_relation(overall_avg_l, avg_pck_scores, save_path=None):
    """
    Plots the overall average brightness against the average PCK scores for each threshold.
    """
    pck_thresholds = [col[-4:] for col in avg_pck_scores.keys()]
    pck_values = list(avg_pck_scores.values())

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel('LOWER PCK Threshold')
    ax1.set_ylabel('Average LOWER PCK Score', color='teal')
    ax1.tick_params(axis='y', labelcolor='teal')

    bars = ax1.bar(pck_thresholds, pck_values, color=['skyblue', 'salmon', 'lightgreen'],
                   label='Average LOWER PCK Score', width=0.6)

    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2f}',
                 ha='center', va='bottom', color='black')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Overall Average Brightness (L)', color='darkred')
    ax2.tick_params(axis='y', labelcolor='darkred')
    ax2.axhline(y=overall_avg_l, color='darkred', linestyle='--', linewidth=2)
    ax2.text(len(pck_thresholds) - 1.25, overall_avg_l, f'{overall_avg_l:.2f}',
             ha='left', va='center', color='darkred', weight='bold',
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    ax2.set_ylim(0, max(overall_avg_l * 1.2, ax1.get_ylim()[1]))

    fig.suptitle(
        'Overall Average LOWER PCK Scores vs. Overall Average Brightness', fontsize=14)
    fig.tight_layout()
    fig.legend(loc='upper right', bbox_to_anchor=(
        1, 1), bbox_transform=ax1.transAxes)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    if save_path:
        plt.savefig(save_path)
        print(f"Overall relation plot saved to {save_path}")
    plt.close()


def plot_pck_vs_metric(df, x_column, y_column, subject_col, action_col, camera_col, title, x_label, save_path=None):
    """
    Plots a scatter plot of PCK scores vs. a given metric.
    Dynamically handles grouping based on available columns.
    """
    # Increased figure size for better legend visibility
    plt.figure(figsize=(20, 12))

    # Dynamically determine which columns to use for grouping
    grouping_cols = [col for col in [subject_col,
                                     action_col, camera_col] if col is not None]

    if not grouping_cols:
        print("Warning: No grouping columns found. Plotting without group labels.")
        plt.scatter(df[x_column], df[y_column], alpha=0.7)
    else:
        unique_groups = df.groupby(grouping_cols)
        colors = cm.get_cmap('tab20', len(unique_groups))
        i = 0
        for name, group in unique_groups:
            label_parts = []
            if subject_col in grouping_cols:
                label_parts.append(
                    f"S{name[grouping_cols.index(subject_col)]}")
            if action_col in grouping_cols:
                label_parts.append(f"A{name[grouping_cols.index(action_col)]}")
            if camera_col in grouping_cols:
                label_parts.append(f"C{name[grouping_cols.index(camera_col)]}")
            label = ", ".join(label_parts)

            plt.scatter(group[x_column], group[y_column],
                        alpha=0.7, color=colors(i), label=label)
            i += 1
        # Adjusted legend placement and added a title for clarity
        plt.legend(title='Video ID', bbox_to_anchor=(1.02, 1),
                   loc='upper left', borderaxespad=0., ncol=3)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(f'LOWER PCK Score ({y_column[-4:]})')
    plt.grid(True)
    # Use subplots_adjust to make room for the legend on the right
    plt.subplots_adjust(right=0.75)

    if save_path:
        plt.savefig(save_path)
        print(f"Scatter plot saved to {save_path}")
    plt.close()


def plot_pck_vs_brightness_trends(df, x_column, y_column, subject_col, action_col, camera_col, title, x_label, save_path=None):
    """
    Plots PCK scores vs. a given metric, connecting per-frame dots to show trends.
    Dynamically handles grouping based on available columns.
    """
    # Increased figure size for better legend visibility
    plt.figure(figsize=(20, 12))

    grouping_cols = [col for col in [subject_col,
                                     action_col, camera_col] if col is not None]
    if not grouping_cols:
        print("Warning: No grouping columns found. Plotting all data in one line.")
        plt.plot(df[x_column], df[y_column], marker='o',
                 markersize=4, linestyle='-', alpha=0.6)
    else:
        unique_videos = df.groupby(grouping_cols)
        colors = cm.get_cmap('tab20', len(unique_videos))
        i = 0
        for name, group in unique_videos:
            label_parts = []
            if subject_col in grouping_cols:
                label_parts.append(
                    f"S{name[grouping_cols.index(subject_col)]}")
            if action_col in grouping_cols:
                label_parts.append(f"A{name[grouping_cols.index(action_col)]}")
            if camera_col in grouping_cols:
                label_parts.append(f"C{name[grouping_cols.index(camera_col)]}")
            label = ", ".join(label_parts)

            plt.plot(group[x_column], group[y_column],
                     marker='o', markersize=4, linestyle='-',
                     color=colors(i), label=label, alpha=0.6)
            i += 1
        # Adjusted legend placement and added a title for clarity
        plt.legend(title='Video ID', bbox_to_anchor=(1.02, 1),
                   loc='upper left', borderaxespad=0., ncol=3)

    x_min_data = df[x_column].min()
    x_max_data = df[x_column].max()
    x_min = np.floor(x_min_data / 5) * 5
    x_max = np.ceil(x_max_data / 5) * 5

    plt.xlim(x_min, x_max)
    plt.xticks(np.arange(x_min, x_max + 1, 5))
    plt.tick_params(axis='x', rotation=45)

    y_min, y_max = df[y_column].min(), df[y_column].max()
    plt.ylim(y_min - (y_max-y_min)*0.05, y_max + (y_max-y_min)*0.05)
    plt.yticks(np.arange(y_min, y_max + 1, 5))
    plt.tick_params(axis='y', rotation=45)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(f'LOWER PCK Score ({y_column[-4:]})')
    plt.grid(True)
    # Use subplots_adjust to make room for the legend on the right
    plt.subplots_adjust(right=0.75)

    if save_path:
        plt.savefig(save_path)
        print(f"Plot with trends saved to {save_path}")
    plt.close()


def plot_pck_vs_brightness_interactive(df, x_column, y_column, subject_col, action_col, camera_col, frame_col, title, save_dir):
    """
    Interactive PCK vs brightness plot.
    Dynamically handles the video identifier.
    """
    grouping_cols = [col for col in [subject_col,
                                     action_col, camera_col] if col is not None]

    # Create a new column combining video info
    df['video_id_label'] = df.apply(lambda row: "_".join(
        [f"{col}:{row[col]}" for col in grouping_cols]), axis=1)

    hover_data = {
        col: True for col in grouping_cols + [frame_col, x_column, y_column]
    }

    fig = px.scatter(
        df,
        x=x_column,
        y=y_column,
        color='video_id_label',
        hover_data=hover_data,
        title=title,
        labels={x_column: "Brightness (L*)",
                y_column: f"LOWER PCK Score ({y_column[-4:]})"}
    )

    fig.update_traces(marker=dict(size=6, opacity=0.8))
    fig.update_layout(template="plotly_white", showlegend=True)
    fig.update_layout(coloraxis_showscale=False)

    html_path = os.path.join(save_dir, f"{title.replace(' ', '_')}.html")
    fig.write_html(html_path)
    print(f"Interactive plot saved to: {html_path}")

    # fig.show()

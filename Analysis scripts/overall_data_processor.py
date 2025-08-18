import os
import pandas as pd
from video_analyzer import VideoAnalyzer
from video_file_mapper import find_video_for_pck_row


def get_overall_data(config, pck_df, metric_name, analyzer_method):
    """
    Processes all videos to extract an overall average for a specified metric,
    and returns a combined DataFrame with PCK scores.

    Args:
        config (object): A configuration object containing dataset-specific parameters.
        pck_df (pd.DataFrame): The DataFrame with overall PCK scores.
        metric_name (str): The name of the metric being analyzed (e.g., 'brightness').
        analyzer_method (str): The name of the VideoAnalyzer method to call
                               (e.g., 'get_brightness_data').

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: The merged DataFrame with PCK and average metric scores.
            - list: A list of all per-frame data points for the metric.
    """
    print("\n" + "=" * 50)
    print(f"Processing video data for {metric_name}...")

    all_metric_data = []
    video_metrics_rows = []

    grouping_cols = [col for col in [config.SUBJECT_COLUMN,
                                     config.ACTION_COLUMN, config.CAMERA_COLUMN] if col is not None]

    if not grouping_cols:
        print("Warning: No grouping columns found in config. Cannot perform per-video analysis.")
        return pd.DataFrame(), []

    grouped_pck_df = pck_df.groupby(grouping_cols)

    for group_name, group_data in grouped_pck_df:
        video_row_data = {
            col: group_name[grouping_cols.index(col)] for col in grouping_cols
        }
        video_row = pd.Series(video_row_data)

        video_path = find_video_for_pck_row(config, video_row)
        if not (video_path and os.path.exists(video_path)):
            video_info = ', '.join(
                [f"{k}: {v}" for k, v in video_row_data.items()])
            print(f"Warning: Video not found for {video_info}. Skipping.")
            continue

        analyzer = VideoAnalyzer(video_path)

        # Use getattr to call the specified method dynamically
        metric_data = getattr(analyzer, analyzer_method, None)()

        if metric_data:
            synced_start_frame = 0
            if hasattr(config, 'sync_data'):
                try:
                    subject_key = f"{video_row_data.get(config.SUBJECT_COLUMN)}"
                    action_key = video_row_data.get(config.ACTION_COLUMN)
                    if isinstance(action_key, str):
                        action_key = action_key.replace('_', ' ').title()

                    camera_id = video_row_data.get(config.CAMERA_COLUMN)
                    camera_index = int(camera_id) - 1

                    synced_start_frame = config.sync_data.data[subject_key][action_key][camera_index]

                    if synced_start_frame < 0:
                        print(
                            f"Warning: Invalid synced start frame {synced_start_frame}. Using frame 0.")
                        synced_start_frame = 0
                except (KeyError, IndexError, TypeError) as e:
                    print(
                        f"Warning: No sync data found for {video_row_data}. Using frame 0. Error: {e}")

            metric_data_sliced = metric_data[synced_start_frame:]
            all_metric_data.extend(metric_data_sliced)

            new_row = video_row_data.copy()
            new_row[f'avg_{metric_name}'] = pd.Series(
                metric_data_sliced).mean()
            video_metrics_rows.append(new_row)

    if not video_metrics_rows:
        print(
            f"No video data could be successfully processed for {metric_name} analysis.")
        return pd.DataFrame(), []

    video_metrics_df = pd.DataFrame(video_metrics_rows)
    merged_df = pd.merge(pck_df, video_metrics_df,
                         on=grouping_cols, how='inner')

    return merged_df, all_metric_data

import os
import pandas as pd
from video_analyzer import VideoAnalyzer
from video_file_mapper import find_video_for_pck_row


def get_per_frame_data(config, pck_per_frame_df, metrics_to_extract):
    """
    Processes all videos to extract per-frame data for specified metrics,
    aligns it with PCK scores, and returns a single combined DataFrame.

    Args:
        config (object): A configuration object with dataset parameters.
        pck_per_frame_df (pd.DataFrame): The DataFrame with per-frame PCK scores.
        metrics_to_extract (dict): A dictionary mapping metric names (e.g., 'brightness')
                                   to the corresponding method names in VideoAnalyzer
                                   (e.g., 'get_brightness_data').

    Returns:
        pd.DataFrame: A DataFrame with combined PCK and all specified per-frame data.
    """
    print("\n" + "=" * 50)
    print("Processing per-frame data...")
    all_per_frame_data = []

    # Dynamically determine the columns to use for grouping
    grouping_cols = [col for col in [config.SUBJECT_COLUMN,
                                     config.ACTION_COLUMN, config.CAMERA_COLUMN] if col is not None]

    if not grouping_cols:
        print("Warning: No grouping columns found in config. Cannot perform per-video analysis.")
        return pd.DataFrame()

    grouped_pck_df = pck_per_frame_df.groupby(grouping_cols)

    for group_name, video_pck_df in grouped_pck_df:
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

        extracted_data = {}
        extraction_failed = False
        for metric_name, method_name in metrics_to_extract.items():
            method = getattr(analyzer, method_name, None)
            if method:
                extracted_data[metric_name] = method()
            else:
                print(
                    f"Error: VideoAnalyzer has no method named '{method_name}'.")
                extraction_failed = True
                break

        if extraction_failed or not all(extracted_data.values()):
            print(
                f"Warning: Could not get all specified data for {video_path}. Skipping.")
            continue

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

        # Slice all extracted data based on the synchronized start frame
        sliced_data = {
            metric: data[synced_start_frame:] for metric, data in extracted_data.items()
        }

        # Determine the minimum length for all data streams to ensure alignment
        lengths = [len(video_pck_df)] + [len(d) for d in sliced_data.values()]
        min_length = min(lengths)

        if len(video_pck_df) != min_length:
            print(
                f"Frame count mismatch for {video_path} after slicing. Using {min_length} frames.")

        video_pck_df_aligned = video_pck_df.head(min_length).copy()

        # Add the aligned metric data to the dataframe
        for metric, data in sliced_data.items():
            video_pck_df_aligned[metric] = data[:min_length]

        video_pck_df_aligned['frame_idx'] = range(
            synced_start_frame, synced_start_frame + min_length)

        for col, value in video_row_data.items():
            video_pck_df_aligned[col] = value

        all_per_frame_data.append(video_pck_df_aligned)

    if not all_per_frame_data:
        print("No video data could be successfully processed and matched with PCK scores.")
        return pd.DataFrame()

    combined_df = pd.concat(all_per_frame_data, ignore_index=True)
    return combined_df

# video_file_mapper.py

import os
import re


def find_video_for_pck_row(config, pck_row):
    """
    Finds a video file based on the metadata in a PCK score DataFrame row,
    handling different directory structures for different datasets.

    Args:
        config (object): The configuration object for the current dataset.
        pck_row (pd.Series): A row from the PCK scores DataFrame.

    Returns:
        str: The full path to the video file, or None if not found.
    """
    # Create a list of the columns we want to use for the pattern
    pattern_cols = [config.SUBJECT_COLUMN,
                    config.ACTION_COLUMN, config.CAMERA_COLUMN]

    # Get values from the row, only for existing columns
    pattern_values = [pck_row.get(col)
                      for col in pattern_cols if col is not None]

    if not pattern_values:
        print("Warning: No video identifier columns found in the config.")
        return None

    # Handle the specific file structure of the Humaneva dataset
    if "humaneva" in config.DATASET_NAME.lower():
        # Humaneva paths are structured like: S1/Image_Data/Walking_1_(C1).avi
        subject_id = pck_row.get(config.SUBJECT_COLUMN)
        action_name = pck_row.get(config.ACTION_COLUMN)
        camera_id = pck_row.get(config.CAMERA_COLUMN)
        camera_id = camera_id+1

        if all([subject_id, action_name, camera_id]):
            # Construct the video filename and path
            # The action name often contains the number, e.g., 'Walking_1'
            filename = f"{action_name}_(C{camera_id}).avi"
            video_path = os.path.join(
                config.VIDEO_DIRECTORY, f"{subject_id}", "Image_Data", filename)

            if os.path.exists(video_path):
                return video_path
            else:
                print(f"Warning: Humaneva video not found at: {video_path}")
                return None
        else:
            print(
                "Warning: Missing subject, action, or camera information for Humaneva dataset.")
            return None

    # Fallback to a generic search for other datasets like MoVi
    else:
        video_pattern = "_".join([str(v) for v in pattern_values])

        # Use a regex to be more flexible with file extensions
        video_regex = re.compile(
            rf'^{video_pattern}.*\.(mp4|avi|mov)$', re.IGNORECASE)

        for root, _, files in os.walk(config.VIDEO_DIRECTORY):
            for f in files:
                if video_regex.match(f):
                    return os.path.join(root, f)

        return None

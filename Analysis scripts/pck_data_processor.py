# pck_data_processor.py

import pandas as pd


class PCKDataProcessor:
    """
    Handles loading and processing PCK score data from a spreadsheet.
    It is designed to be generic, accepting a configuration object to
    handle dataset-specific column names and sheet names.
    """

    def __init__(self, config):
        """
        Initializes the processor with the configuration object.

        Args:
            config (object): A configuration object containing file paths and column names.
        """
        self.config = config

    def load_pck_scores(self):
        """
        Loads the PCK scores from the 'Overall Metrics' sheet into a pandas DataFrame.
        Dynamically handles required columns based on the config.

        Returns:
            pd.DataFrame or None: The DataFrame with PCK scores, or None if loading fails.
        """
        try:
            df = pd.read_excel(self.config.PCK_FILE_PATH,
                               sheet_name='Overall Metrics', header=0)

            # Dynamically determine the video identifier columns from the config
            id_cols = [col for col in [self.config.SUBJECT_COLUMN,
                                       self.config.ACTION_COLUMN, self.config.CAMERA_COLUMN] if col is not None]

            # Drop rows where any of the identifier columns are missing
            if id_cols:
                df = df.dropna(subset=id_cols).reset_index(drop=True)

            # Dynamically determine all required columns
            required_cols = id_cols + self.config.PCK_OVERALL_SCORE_COLUMNS
            if not all(col in df.columns for col in required_cols):
                print(f"Error: One or more required columns are missing from the 'Overall Metrics' sheet. "
                      f"Expected: {required_cols}")
                return None

            # Handle the camera column data type if it exists
            if self.config.CAMERA_COLUMN in df.columns:
                df[self.config.CAMERA_COLUMN] = df[self.config.CAMERA_COLUMN].astype(
                    int)

            return df
        except FileNotFoundError:
            print(
                f"Error: The file {self.config.PCK_FILE_PATH} was not found.")
            return None
        except Exception as e:
            print(
                f"An error occurred while loading the 'Overall Metrics' sheet: {e}")
            return None

    def load_pck_per_frame_scores(self):
        """
        Loads the per-frame PCK scores from the 'Per-Frame Scores' sheet.
        Dynamically handles required columns based on the config.

        Returns:
            pd.DataFrame or None: The DataFrame with per-frame scores, or None if loading fails.
        """
        try:
            df = pd.read_excel(self.config.PCK_FILE_PATH,
                               sheet_name='Per-Frame Scores', header=0)

            # Dynamically determine all required columns
            id_cols = [col for col in [self.config.SUBJECT_COLUMN,
                                       self.config.ACTION_COLUMN, self.config.CAMERA_COLUMN] if col is not None]
            required_cols = id_cols + ['frame_idx'] + \
                self.config.PCK_PER_FRAME_SCORE_COLUMNS

            if not all(col in df.columns for col in required_cols):
                print(f"Error: One or more required columns are missing from the 'Per-Frame Scores' sheet. "
                      f"Expected: {required_cols}")
                return None

            # Handle the camera column data type if it exists
            if self.config.CAMERA_COLUMN in df.columns:
                df[self.config.CAMERA_COLUMN] = df[self.config.CAMERA_COLUMN].astype(
                    int)

            return df
        except FileNotFoundError:
            print(
                f"Error: The file {self.config.PCK_FILE_PATH} was not found.")
            return None
        except Exception as e:
            print(
                f"An error occurred while loading the 'Per-Frame Scores' sheet: {e}")
            return None

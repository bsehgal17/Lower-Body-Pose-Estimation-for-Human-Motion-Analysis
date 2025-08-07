import pandas as pd


class PCKDataProcessor:
    """
    Handles loading and processing PCK score data from an Excel file.
    """

    def __init__(self, file_path, score_columns):
        """
        Initializes the processor with the file path and relevant columns.

        Args:
            file_path (str): Path to the Excel file.
            score_columns (list): List of column names for PCK scores.
        """
        self.file_path = file_path
        self.score_columns = score_columns

    def load_pck_scores(self):
        """
        Loads the PCK scores from the Excel file into a pandas DataFrame.
        Assumes the first row contains the column headers.

        Returns:
            pd.DataFrame or None: The DataFrame with PCK scores, or None if loading fails.
        """
        try:
            # Load the data, assuming the first row is the header.
            df = pd.read_excel(self.file_path, header=0)

            # --- New Logic: Clean the DataFrame ---
            # Remove any rows that have NaN in the 'subject' column,
            # as these are likely summary rows and not per-video data points.
            df = df.dropna(subset=['subject']).reset_index(drop=True)

            # Ensure the required columns exist
            required_cols = ['subject', 'action',
                             'camera'] + self.score_columns
            if not all(col in df.columns for col in required_cols):
                print(f"Error: One or more required columns are missing from the Excel file. "
                      f"Expected: {required_cols}")
                return None

            # The 'subject' column is read as a string, but the 'camera' might be an int.
            # This ensures consistent data types for processing.
            df['camera'] = df['camera'].astype(int)
            return df
        except FileNotFoundError:
            print(f"Error: The file {self.file_path} was not found.")
            return None
        except Exception as e:
            print(f"An error occurred while loading the Excel file: {e}")
            return None

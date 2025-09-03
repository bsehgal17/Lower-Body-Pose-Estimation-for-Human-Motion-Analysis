"""
File I/O utilities.
"""

import os
import pandas as pd
from typing import Optional


class FileUtils:
    """Utilities for file operations."""

    @staticmethod
    def ensure_directory_exists(file_path: str) -> None:
        """Ensure the directory for a file path exists."""
        directory = os.path.dirname(file_path)
        os.makedirs(directory, exist_ok=True)

    @staticmethod
    def save_dataframe_to_excel(
        df: pd.DataFrame,
        file_path: str,
        append_if_exists: bool = True,
        sheet_name: str = "Sheet1",
    ) -> None:
        """Save DataFrame to Excel with optional append functionality."""
        FileUtils.ensure_directory_exists(file_path)

        if append_if_exists and os.path.exists(file_path):
            try:
                existing_df = pd.read_excel(file_path, sheet_name=sheet_name)
                combined_df = pd.concat([existing_df, df], ignore_index=True)
                combined_df.to_excel(file_path, sheet_name=sheet_name, index=False)
                print(f"Updated data saved to '{file_path}'")
            except Exception as e:
                print(f"Error appending to existing file: {e}")
                df.to_excel(file_path, sheet_name=sheet_name, index=False)
                print(f"New file created at '{file_path}'")
        else:
            df.to_excel(file_path, sheet_name=sheet_name, index=False)
            print(f"Data saved to '{file_path}'")

    @staticmethod
    def load_excel_safely(
        file_path: str, sheet_name: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """Safely load Excel file with error handling."""
        try:
            if sheet_name:
                return pd.read_excel(file_path, sheet_name=sheet_name)
            else:
                return pd.read_excel(file_path)
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            return None
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            return None

"""
PCK data loading and processing.
"""

import pandas as pd
from typing import List
from core.base_classes import BaseDataProcessor
from utils.data_validator import DataValidator


class PCKDataLoader(BaseDataProcessor):
    """Loads PCK scores from Excel files."""

    def load_overall_scores(self) -> pd.DataFrame:
        """Load overall PCK scores from Excel file."""
        return self._load_excel_sheet(
            "Overall Metrics", self.config.pck_overall_score_columns
        )

    def load_per_frame_scores(self) -> pd.DataFrame:
        """Load per-frame PCK scores from Excel file."""
        required_cols = ["frame_idx"] + self.config.pck_per_frame_score_columns
        return self._load_excel_sheet("Per-Frame Scores", required_cols)

    def load_jointwise_scores(self) -> pd.DataFrame:
        """Load jointwise PCK scores from Excel file."""
        return self._load_excel_sheet(
            "Jointwise Metrics", self.config.pck_jointwise_score_columns
        )

    def _load_excel_sheet(
        self, sheet_name: str, required_score_columns: List[str]
    ) -> pd.DataFrame:
        """Generic method to load Excel sheets with validation."""
        try:
            df = pd.read_excel(
                self.config.pck_file_path, sheet_name=sheet_name, header=0
            )

            grouping_cols = self.config.get_grouping_columns()

            if grouping_cols:
                df = df.dropna(subset=grouping_cols)
                df = df.reset_index(drop=True)

            required_cols = grouping_cols + required_score_columns
            if not DataValidator.validate_required_columns(
                df, required_cols, f"{sheet_name} sheet"
            ):
                return None

            if self.config.camera_column and self.config.camera_column in df.columns:
                df[self.config.camera_column] = df[self.config.camera_column].astype(
                    int
                )

            print(f"Successfully loaded {len(df)} records from '{sheet_name}' sheet")
            return df

        except FileNotFoundError:
            print(f"Error: The file {self.config.pck_file_path} was not found.")
            return None
        except Exception as e:
            print(f"An error occurred while loading the '{sheet_name}' sheet: {e}")
            return None

    def process(self, *args, **kwargs) -> pd.DataFrame:
        """Main processing method."""
        sheet_type = kwargs.get("sheet_type", "overall")
        if sheet_type == "overall":
            return self.load_overall_scores()
        elif sheet_type == "per_frame":
            return self.load_per_frame_scores()
        elif sheet_type == "jointwise":
            return self.load_jointwise_scores()
        else:
            raise ValueError(f"Unknown sheet type: {sheet_type}")

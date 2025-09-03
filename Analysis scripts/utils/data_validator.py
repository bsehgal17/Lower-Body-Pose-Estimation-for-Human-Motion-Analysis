"""
Data validation utilities.
"""

import pandas as pd
from typing import List


class DataValidator:
    """Utilities for data validation and cleaning."""

    @staticmethod
    def validate_required_columns(
        df: pd.DataFrame, required_columns: List[str], data_name: str = "data"
    ) -> bool:
        """Validate that required columns exist in DataFrame."""
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Error in {data_name}: Missing required columns: {missing_columns}")
            return False
        return True

    @staticmethod
    def clean_missing_values(
        df: pd.DataFrame,
        columns_to_clean: List[str],
        fill_value: float = 0.0,
        drop_rows_with_na: bool = True,
    ) -> pd.DataFrame:
        """Clean missing values in specified columns."""
        cleaned_df = df.copy()

        if cleaned_df.isnull().values.any():
            print(f"Warning: Missing values found. Filling with {fill_value}...")
            cleaned_df.loc[:, columns_to_clean] = cleaned_df.loc[
                :, columns_to_clean
            ].fillna(fill_value)

            if drop_rows_with_na:
                cleaned_df.dropna(inplace=True)

        return cleaned_df

    @staticmethod
    def validate_data_ranges(
        df: pd.DataFrame, column_ranges: dict, data_name: str = "data"
    ) -> bool:
        """Validate that data falls within expected ranges."""
        errors = []

        for column, (min_val, max_val) in column_ranges.items():
            if column not in df.columns:
                continue

            out_of_range = df[(df[column] < min_val) | (df[column] > max_val)]

            if not out_of_range.empty:
                errors.append(
                    f"Column '{column}' has {len(out_of_range)} values outside range [{min_val}, {max_val}]"
                )

        if errors:
            print(f"Data range validation errors in {data_name}:")
            for error in errors:
                print(f"  - {error}")
            return False

        return True

    @staticmethod
    def check_data_completeness(df: pd.DataFrame, data_name: str = "data") -> dict:
        """Check data completeness and return statistics."""
        stats = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "missing_values_per_column": df.isnull().sum().to_dict(),
            "missing_percentage_per_column": (
                df.isnull().sum() / len(df) * 100
            ).to_dict(),
            "complete_rows": len(df.dropna()),
            "completeness_percentage": len(df.dropna()) / len(df) * 100
            if len(df) > 0
            else 0,
        }

        print(f"Data completeness report for {data_name}:")
        print(f"  Total rows: {stats['total_rows']}")
        print(
            f"  Complete rows: {stats['complete_rows']} ({stats['completeness_percentage']:.1f}%)"
        )

        columns_with_missing = [
            col
            for col, count in stats["missing_values_per_column"].items()
            if count > 0
        ]
        if columns_with_missing:
            print(f"  Columns with missing values: {columns_with_missing}")

        return stats

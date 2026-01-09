"""
Script to aggregate PCK evaluation results from Excel files.

Reads evaluation Excel files from folder structure, extracts:
- Joint names and thresholds from column names (e.g., LEFT_HIP_jointwise_pck_0.02)
- Frequency from folder path (e.g., cutoff5.0)
- Averages PCK values across all rows

Usage:
1. Set ROOT_PATH to a folder containing Excel evaluation files
2. Frequency will be extracted from folder path (looks for cutoff<value>)
3. Script recursively searches for *.xlsx and *.xls files in all subfolders
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class EvaluationExcelAggregator:
    """Aggregates PCK results from evaluation Excel files."""

    def __init__(self, root_path: str):
        """
        Initialize the aggregator.

        Args:
            root_path: Root directory containing pipeline results with Excel files
        """
        self.root_path = Path(root_path)
        self.results = []  # List of dicts for DataFrame

        logger.info(f"Initialized EvaluationExcelAggregator")
        logger.info(f"Root Path: {self.root_path}")

    def extract_frequency_from_path(self, path: str) -> Optional[float]:
        """
        Extract filter frequency from folder name.
        E.g., 'butterworth_order1_cutoff5.0_fs60.0' -> 5.0
        """
        match = re.search(r"cutoff(\d+\.?\d*)", path)
        if match:
            return float(match.group(1))
        return None

    def parse_column_name(self, col_name: str) -> Optional[Tuple[str, float]]:
        """
        Parse column name to extract joint name and threshold.
        E.g., 'LEFT_HIP_jointwise_pck_0.02' -> ('LEFT_HIP', 0.02)

        Args:
            col_name: Column name from Excel

        Returns:
            Tuple of (joint_name, threshold) or None if not a PCK column
        """
        # Pattern: JOINT_NAME_jointwise_pck_THRESHOLD
        match = re.match(r"(.+?)_jointwise_pck_([\d.]+)$", col_name)
        if match:
            joint_name = match.group(1)
            threshold = float(match.group(2))
            return (joint_name, threshold)
        return None

    def find_evaluation_files(self) -> List[Tuple[Path, float]]:
        """
        Find all evaluation Excel files recursively in folder structure.
        Extracts frequency from the folder path.

        Returns:
            List of (file_path, frequency) tuples
        """
        files = []

        # Recursively find all Excel files in the root path and subfolders
        excel_files = list(self.root_path.glob("**/*.xlsx")) + list(
            self.root_path.glob("**/*.xls")
        )
        logger.info(f"Found {len(excel_files)} Excel files total")

        for excel_file in excel_files:
            # Extract frequency from the full path
            path_str = str(excel_file)
            frequency = self.extract_frequency_from_path(path_str)

            if frequency is not None:
                files.append((excel_file, frequency))
                logger.debug(f"Found Excel file: {excel_file.name} (freq={frequency})")
            else:
                logger.warning(
                    f"Could not extract frequency from path: {excel_file.parent.name}"
                )

        logger.info(f"Found {len(files)} evaluation Excel files with frequency")
        return files

    def process_excel_file(self, excel_path: Path, frequency: float) -> List[Dict]:
        """
        Process a single evaluation Excel file.

        Args:
            excel_path: Path to Excel file
            frequency: Extracted frequency value

        Returns:
            List of result dictionaries with joint, threshold, avg_pck
        """
        results = []

        try:
            # Read all sheets and combine
            xl_file = pd.ExcelFile(excel_path)
            all_data = pd.concat(
                [
                    pd.read_excel(excel_path, sheet_name=sheet)
                    for sheet in xl_file.sheet_names
                ],
                ignore_index=True,
            )

            # Find PCK columns
            pck_columns = {}  # {col_name: (joint, threshold)}

            for col in all_data.columns:
                parsed = self.parse_column_name(col)
                if parsed:
                    joint_name, threshold = parsed
                    pck_columns[col] = (joint_name, threshold)

            if not pck_columns:
                logger.warning(f"No PCK columns found in {excel_path.name}")
                return results

            # Extract average for each joint-threshold combination
            for col, (joint, threshold) in pck_columns.items():
                try:
                    # Get numeric values and compute average
                    values = pd.to_numeric(all_data[col], errors="coerce")
                    valid_values = values.dropna()

                    if len(valid_values) > 0:
                        avg_pck = valid_values.mean()
                        results.append(
                            {
                                "Joint": joint,
                                "Frequency": frequency,
                                "Threshold": threshold,
                                "Average PCK (%)": round(avg_pck, 2),
                            }
                        )
                except Exception as e:
                    logger.error(
                        f"Error processing column {col} in {excel_path.name}: {e}"
                    )
                    continue

        except Exception as e:
            logger.error(f"Error reading Excel file {excel_path}: {e}")

        return results

    def process_files(self) -> pd.DataFrame:
        """
        Process all evaluation Excel files and aggregate results.

        Returns:
            DataFrame with columns: Joint, Frequency, Threshold, Average PCK (%)
        """
        excel_files = self.find_evaluation_files()

        if not excel_files:
            logger.warning("No evaluation Excel files found!")
            return pd.DataFrame()

        for excel_path, frequency in excel_files:
            logger.info(f"Processing: {excel_path}")
            file_results = self.process_excel_file(excel_path, frequency)
            self.results.extend(file_results)

        df = pd.DataFrame(self.results)

        # Sort by joint, threshold, then frequency
        if not df.empty:
            df = df.sort_values(["Joint", "Threshold", "Frequency"])

        return df

    def save_excel(self, output_path: str):
        """Save aggregated results to Excel file with different sheet structures."""
        df = self.process_files()

        if df.empty:
            logger.error("No results to save!")
            return

        # Create Excel writer
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            # 1. For each threshold - create sheet with Frequency as rows, Joint as columns
            thresholds = sorted(df["Threshold"].unique())
            for threshold in thresholds:
                threshold_data = df[df["Threshold"] == threshold]
                pivot_table = threshold_data.pivot_table(
                    index="Frequency",
                    columns="Joint",
                    values="Average PCK (%)",
                    aggfunc="mean",
                )
                sheet_name = f"Threshold_{threshold}"
                pivot_table.to_excel(writer, sheet_name=sheet_name)

            # 2. Sheet with Joint as columns, Frequency as rows - Average across all thresholds
            avg_across_threshold = (
                df.groupby(["Frequency", "Joint"])["Average PCK (%)"]
                .mean()
                .unstack(fill_value=None)
            )
            avg_across_threshold.to_excel(writer, sheet_name="Avg Across Thresholds")

            # 3. Sheet with Joint as columns, Frequency as rows - Threshold with highest PCK
            highest_threshold_sheet = []
            for (freq, joint), group in df.groupby(["Frequency", "Joint"]):
                max_idx = group["Average PCK (%)"].idxmax()
                best_threshold = group.loc[max_idx, "Threshold"]
                highest_threshold_sheet.append(
                    {
                        "Frequency": freq,
                        "Joint": joint,
                        "Best Threshold": best_threshold,
                    }
                )

            best_threshold_df = pd.DataFrame(highest_threshold_sheet)
            best_threshold_pivot = best_threshold_df.pivot_table(
                index="Frequency",
                columns="Joint",
                values="Best Threshold",
                aggfunc="first",
            )
            best_threshold_pivot.to_excel(writer, sheet_name="Best Threshold")

            # Format worksheets
            workbook = writer.book

            for sheet_name in workbook.sheetnames:
                worksheet = writer.sheets[sheet_name]
                for column in worksheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                    adjusted_width = min(max_length + 2, 50)
                    worksheet.column_dimensions[column_letter].width = adjusted_width

        logger.info(f"Results saved to {output_path}")
        logger.info(f"Total records: {len(df)}")
        print("\n" + "=" * 80)
        print("Summary of created sheets:")
        print(
            f"  - Individual sheets for each threshold (rows: Frequency, cols: Joint)"
        )
        print(f"  - Avg Across Thresholds (rows: Frequency, cols: Joint)")
        print(f"  - Best Threshold (rows: Frequency, cols: Joint)")
        print("=" * 80)


def main():
    """Main execution function - configure parameters below."""
    import sys

    # ============================================================================
    # CONFIGURATION - Edit these parameters before running
    # ============================================================================

    # Root path containing pipeline results with evaluation Excel files
    ROOT_PATH = r"C:\path\to\pipeline_results\HumanSC3D"

    # Output Excel file path for aggregated results
    OUTPUT_PATH = "pck_evaluation_summary.xlsx"

    # ============================================================================

    print("\n" + "=" * 80)
    print("PCK Evaluation Results Aggregator")
    print("=" * 80)

    print(f"\nConfiguration:")
    print(f"  Root Path: {ROOT_PATH}")
    print(f"  Output Path: {OUTPUT_PATH}")
    print("-" * 80 + "\n")

    if not Path(ROOT_PATH).exists():
        print(f"✗ Error: Path does not exist: {ROOT_PATH}")
        sys.exit(1)

    try:
        aggregator = EvaluationExcelAggregator(ROOT_PATH)
        aggregator.save_excel(OUTPUT_PATH)
        print(f"\n✓ Successfully generated report: {OUTPUT_PATH}")
    except Exception as e:
        print(f"\n✗ Error during processing: {e}")
        logger.exception("Detailed error:")
        sys.exit(1)


if __name__ == "__main__":
    main()

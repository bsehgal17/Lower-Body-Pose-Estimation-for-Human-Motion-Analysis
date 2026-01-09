"""
PCK Results Aggregator - Workflow:
1. Find Excel files in folder and subfolders, extract frequency from path
2. For each Excel file, parse columns like LEFT_ELBOW_jointwise_pck_0.03
3. Group columns by joint name, average PCK across all thresholds for unified value
4. Create sheets: (Frequency, Video) -> Columns: Joints, Values: Avg PCK across thresholds
5. Create summary sheet: Columns: (Video, Joint), Values: Best frequency where avg PCK is highest
"""

import re
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PCKAggregator:
    """Aggregates PCK results from evaluation Excel files."""

    def __init__(self, root_path: str):
        self.root_path = Path(root_path)
        self.results = []
        logger.info(f"Initialized PCKAggregator with root: {self.root_path}")

    def extract_frequency_from_path(self, path: str) -> Optional[float]:
        """Extract filter frequency from folder name. E.g., cutoff5.0 -> 5.0"""
        match = re.search(r"cutoff(\d+\.?\d*)", path)
        if match:
            return float(match.group(1))
        return None

    def parse_joint_name(self, col_name: str) -> Optional[str]:
        """
        Extract joint name from column name.
        E.g., 'LEFT_HIP_jointwise_pck_0.02' -> 'LEFT_HIP'
        """
        match = re.match(r"(.+?)_jointwise_pck_([\d.]+)$", col_name)
        if match:
            return match.group(1)
        return None

    def get_video_name(self, excel_file: Path) -> str:
        """Extract video name from Excel filename (without extension)."""
        return excel_file.stem

    def find_evaluation_files(self) -> List[Tuple[Path, float]]:
        """
        Find all evaluation Excel files recursively.
        Returns list of (file_path, frequency) tuples
        """
        files = []
        excel_files = list(self.root_path.glob("**/*.xlsx")) + list(
            self.root_path.glob("**/*.xls")
        )
        logger.info(f"Found {len(excel_files)} Excel files total")

        for excel_file in excel_files:
            path_str = str(excel_file)
            frequency = self.extract_frequency_from_path(path_str)

            if frequency is not None:
                files.append((excel_file, frequency))
                logger.debug(f"Found: {excel_file.name} (freq={frequency})")
            else:
                logger.warning(f"Could not extract frequency from: {excel_file}")

        logger.info(f"Found {len(files)} Excel files with valid frequency")
        return files

    def process_excel_file(self, excel_path: Path, frequency: float) -> List[Dict]:
        """
        Process Excel file:
        - Read all sheets
        - Group columns by joint name
        - Average PCK across all thresholds for each joint
        - Return list of dicts: {video, frequency, joint, avg_pck}
        """
        results = []
        video_name = self.get_video_name(excel_path)

        try:
            xl_file = pd.ExcelFile(excel_path)
            all_data = pd.concat(
                [
                    pd.read_excel(excel_path, sheet_name=sheet)
                    for sheet in xl_file.sheet_names
                ],
                ignore_index=True,
            )

            # Group columns by joint name
            joint_columns = {}  # {joint_name: [col1, col2, ...]}

            for col in all_data.columns:
                joint_name = self.parse_joint_name(col)
                if joint_name:
                    if joint_name not in joint_columns:
                        joint_columns[joint_name] = []
                    joint_columns[joint_name].append(col)

            if not joint_columns:
                logger.warning(f"No PCK columns found in {excel_path.name}")
                return results

            # For each joint, average across all threshold columns
            for joint_name, columns in joint_columns.items():
                # Convert all columns to numeric
                numeric_arrays = []
                for col in columns:
                    values = pd.to_numeric(all_data[col], errors="coerce")
                    numeric_arrays.append(values)

                # Stack and compute mean across all thresholds and all rows
                combined_df = pd.concat(numeric_arrays, axis=1)
                avg_pck = combined_df.values.flatten()
                avg_pck = avg_pck[~pd.isna(avg_pck)].mean()

                results.append(
                    {
                        "Video": video_name,
                        "Frequency": frequency,
                        "Joint": joint_name,
                        "Average PCK (%)": round(avg_pck, 2),
                    }
                )

        except Exception as e:
            logger.error(f"Error reading {excel_path}: {e}")

        return results

    def process_files(self) -> pd.DataFrame:
        """Process all Excel files and return aggregated DataFrame."""
        excel_files = self.find_evaluation_files()

        if not excel_files:
            logger.warning("No Excel files found!")
            return pd.DataFrame()

        for excel_path, frequency in excel_files:
            logger.info(f"Processing: {excel_path}")
            file_results = self.process_excel_file(excel_path, frequency)
            self.results.extend(file_results)

        df = pd.DataFrame(self.results)

        if not df.empty:
            df = df.sort_values(["Video", "Frequency", "Joint"])

        return df

    def save_excel(self, output_path: str):
        """
        Create Excel file with:
        1. Separate sheets for each (frequency, video) pair
           - Columns: Joint names
           - Values: Avg PCK across all thresholds
        2. Summary sheet: Best frequency for each (video, joint) combination
        """
        df = self.process_files()

        if df.empty:
            logger.error("No results to save!")
            return

        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            # 1. Create sheets for each (frequency, video) combination
            freq_video_pairs = (
                df.groupby(["Frequency", "Video"])
                .size()
                .reset_index(name="count")[["Frequency", "Video"]]
            )

            for _, row in freq_video_pairs.iterrows():
                frequency = row["Frequency"]
                video = row["Video"]

                # Filter data for this frequency and video
                subset = df[(df["Frequency"] == frequency) & (df["Video"] == video)]

                # Create pivot: rows=video, columns=joints, values=avg_pck
                pivot = subset.pivot_table(
                    index="Video",
                    columns="Joint",
                    values="Average PCK (%)",
                    aggfunc="first",
                )

                # Create sheet name with frequency and video
                sheet_name = f"F{frequency}_{video}"[:31]  # Excel sheet name limit
                pivot.to_excel(writer, sheet_name=sheet_name)
                logger.debug(f"Created sheet: {sheet_name}")

            # 2. Create summary sheet: best frequency for each (video, joint)
            summary_data = []
            for (video, joint), group in df.groupby(["Video", "Joint"]):
                max_idx = group["Average PCK (%)"].idxmax()
                best_freq = group.loc[max_idx, "Frequency"]
                summary_data.append(
                    {
                        "Video": video,
                        "Joint": joint,
                        "Best Frequency": best_freq,
                    }
                )

            summary_df = pd.DataFrame(summary_data)
            summary_pivot = summary_df.pivot_table(
                index="Video", columns="Joint", values="Best Frequency", aggfunc="first"
            )
            summary_pivot.to_excel(writer, sheet_name="Best Frequency")

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
        print("\n" + "=" * 80)
        print("Excel file generated successfully!")
        print("Sheets created:")
        print(f"  - Individual sheets: Freq<frequency>_<video>")
        print(f"  - Summary sheet: Best Frequency")
        print("=" * 80)


def main():
    """Main execution function - configure parameters below."""
    import sys

    # ============================================================================
    # CONFIGURATION - Edit these parameters before running
    # ============================================================================

    # Root path containing filtered folders with Excel evaluation files
    ROOT_PATH = r"C:\path\to\filtered_folders"

    # Output Excel file path for aggregated results
    OUTPUT_PATH = "pck_summary.xlsx"

    # ============================================================================

    print("\n" + "=" * 80)
    print("PCK Results Aggregator")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Root Path: {ROOT_PATH}")
    print(f"  Output Path: {OUTPUT_PATH}")
    print("-" * 80 + "\n")

    if not Path(ROOT_PATH).exists():
        print(f"✗ Error: Path does not exist: {ROOT_PATH}")
        sys.exit(1)

    try:
        aggregator = PCKAggregator(ROOT_PATH)
        aggregator.save_excel(OUTPUT_PATH)
        print(f"\n✓ Successfully generated report: {OUTPUT_PATH}")
    except Exception as e:
        print(f"\n✗ Error during processing: {e}")
        logger.exception("Detailed error:")
        sys.exit(1)


if __name__ == "__main__":
    main()

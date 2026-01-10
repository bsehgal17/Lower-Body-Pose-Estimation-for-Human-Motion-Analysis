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
from itertools import product

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
        """Extract filter frequency from folder name. E.g., filter_butterworth_18th_14hz -> 14"""
        match = re.search(r"(\d+)hz", path, re.IGNORECASE)
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
        Find all evaluation Excel files recursively in all subfolders.
        Returns list of (file_path, frequency) tuples
        """
        files = []

        # Recursively find all Excel files at any depth
        excel_files = list(self.root_path.glob("**/*.xlsx")) + list(
            self.root_path.glob("**/*.xls")
        )
        logger.info(f"Found {len(excel_files)} Excel files total across all folders")

        for excel_file in excel_files:
            path_str = str(excel_file)
            frequency = self.extract_frequency_from_path(path_str)

            if frequency is not None:
                files.append((excel_file, frequency))
                logger.info(
                    f"Found: {excel_file.relative_to(self.root_path)} (freq={frequency}Hz)"
                )
            else:
                logger.warning(
                    f"Could not extract frequency from: {excel_file.relative_to(self.root_path)}"
                )

        logger.info(f"Total files with valid frequency: {len(files)}")
        return files

    def process_excel_file(self, excel_path: Path, frequency: float) -> List[Dict]:
        """
        Process Excel file:
        - Read all sheets
        - For each row (video), create unique ID from subject/action/camera
        - For each row, group columns by joint name
        - Average PCK across all thresholds for each joint per row
        - Return list of dicts: {video, frequency, joint, avg_pck}
        """
        results = []

        try:
            xl_file = pd.ExcelFile(excel_path)
            all_data = pd.concat(
                [
                    pd.read_excel(excel_path, sheet_name=sheet)
                    for sheet in xl_file.sheet_names
                ],
                ignore_index=True,
            )

            # Identify metadata columns (non-PCK columns like subject, action, camera)
            metadata_cols = [
                col for col in all_data.columns if not self.parse_joint_name(col)
            ]

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

            # Process each row
            for idx, row in all_data.iterrows():
                # Create unique video ID from metadata columns
                video_parts = []
                for col in metadata_cols:
                    val = row[col]
                    # Skip NaN values and numeric totals
                    if pd.notna(val) and val != "":
                        video_parts.append(str(val))

                # Skip rows without video metadata
                if not video_parts:
                    logger.debug(f"Skipping row {idx} - no video metadata found")
                    continue

                video_id = "_".join(video_parts)

                # For each joint, average across all threshold columns for this row
                for joint_name, columns in joint_columns.items():
                    # Get values for all threshold columns of this joint in this row
                    values = []
                    for col in columns:
                        try:
                            val = pd.to_numeric(row[col], errors="coerce")
                            if pd.notna(val):
                                values.append(val)
                        except:
                            pass

                    # Average across thresholds for this joint-video combo
                    if values:
                        avg_pck = sum(values) / len(values)
                        results.append(
                            {
                                "Video": video_id,
                                "Frequency": frequency,
                                "Joint": joint_name,
                                "Average PCK (%)": round(avg_pck, 2),
                            }
                        )

        except Exception as e:
            logger.error(f"Error reading {excel_path}: {e}")
            import traceback

            logger.error(traceback.format_exc())

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
        1. Separate sheets for each frequency
           - Rows: Video names
           - Columns: Joint names
           - Values: Avg PCK across all thresholds
        2. Summary sheet: Best frequency for each (video, joint) combination
        """
        df = self.process_files()

        if df.empty:
            logger.error("No results to save!")
            return

        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            # 1. Create sheets for each frequency
            frequencies = sorted(df["Frequency"].unique())
            logger.info(
                f"Creating sheets for {len(frequencies)} frequencies: {frequencies}"
            )

            for frequency in frequencies:
                # Filter data for this frequency (all videos and joints)
                freq_data = df[df["Frequency"] == frequency]

                # Create pivot: rows=video, columns=joints, values=avg_pck
                pivot = freq_data.pivot_table(
                    index="Video",
                    columns="Joint",
                    values="Average PCK (%)",
                    aggfunc="first",
                )

                # Create sheet name with frequency
                sheet_name = f"Freq_{frequency}Hz"
                pivot.to_excel(writer, sheet_name=sheet_name)
                logger.info(f"Created sheet: {sheet_name} with {len(pivot)} videos")

            # 2. Create summary sheet: all best frequencies for each (video, joint)
            summary_data = []
            for (video, joint), group in df.groupby(["Video", "Joint"]):
                max_pck = group["Average PCK (%)"].max()
                # Get ALL frequencies with the highest PCK
                best_freqs = sorted(
                    group[group["Average PCK (%)"] == max_pck]["Frequency"].unique()
                )
                # Convert list to comma-separated string
                best_freqs_str = ", ".join(
                    [str(int(f)) if f.is_integer() else str(f) for f in best_freqs]
                )

                summary_data.append(
                    {
                        "Video": video,
                        "Joint": joint,
                        "Best Frequencies": best_freqs_str,
                    }
                )

            summary_df = pd.DataFrame(summary_data)
            summary_pivot = summary_df.pivot_table(
                index="Video",
                columns="Joint",
                values="Best Frequencies",
                aggfunc="first",
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
        print(
            f"  - Per-frequency sheets: Freq_<frequency>Hz (rows: Videos, cols: Joints)"
        )
        print(f"  - Summary sheet: Best Frequency")
        print("=" * 80)

    def save_permutations_excel(self, output_path: str):
        """[DEPRECATED - Removed per user request]"""
        pass


def main():
    """Main execution function - configure parameters below."""
    import sys

    # ============================================================================
    # CONFIGURATION - Edit these parameters before running
    # ============================================================================

    # Root path containing filtered folders with Excel evaluation files
    ROOT_PATH = r"/storageh100/Projects/Gaitly/bsehgal/pipeline_results/HumanEva/Butterworth_filter/"

    # Output Excel file path for aggregated results
    OUTPUT_PATH = r"/storage/Projects/Gaitly/bsehgal/lower_body_pose_est/pipeline_results/Adaptive_filt/pck_summary.xlsx"

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
        print(f"\n✓ Successfully generated report:")
        print(f"   - {OUTPUT_PATH}")
    except Exception as e:
        print(f"\n✗ Error during processing: {e}")
        logger.exception("Detailed error:")
        sys.exit(1)


if __name__ == "__main__":
    main()

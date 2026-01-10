"""
Temporal Stability Calculator for Video Pose Data

This script:
1. Reads an Excel file to extract the best frequency for each joint (highest PCK)
2. Processes video JSON files organized by frequency folders
3. Calculates temporal stability (jitter) of X-coordinates for each joint
4. Uses joint enum to map joint names to their indices
5. Generates comprehensive Excel report with stability metrics

How temporal stability is measured (X-coordinate only):
- For each joint trajectory over time: x(1), x(2), x(3), ...
- Compute frame-to-frame changes: Δx = x(t) - x(t-1)
- Calculate standard deviation of Δx: represents smoothness
  * Small std → smooth motion (desired)
  * Large std → jittery motion (noisy)

Usage:
    python calc_stability.py <video_json_folder> <best_freq_excel> <output_excel>

Example:
    python calc_stability.py ./pose_videos ./best_frequencies.xlsx ./stability_report.xlsx
"""

import json
import os
import re
import numpy as np
import pandas as pd
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from openpyxl import Workbook
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from enum import Enum

# Import joint enum
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.joint_enum import (
    GTJointsHumanSC3D,
    GTJointsHumanEVa,
    GTJointsMoVi,
    PredJointsDeepLabCut,
    Mediapipe,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TemporalStabilityCalculator:
    """Calculate temporal stability (jitter) for X-coordinate of joint trajectories from video JSON files."""

    def __init__(
        self, video_json_folder: str, excel_file: str, joint_enum_class: Enum = None
    ):
        """
        Initialize the temporal stability calculator.

        Args:
            video_json_folder: Path to folder containing video pose JSON files (with frequency subfolders)
            excel_file: Path to Excel file with best frequencies per joint
            joint_enum_class: Joint enum class to use for mapping joint names to indices
                            Options:
                            - PredJointsDeepLabCut (for RTMW pose estimations) - DEFAULT
                            - GTJointsHumanSC3D (for ground truth HumanSC3D)
                            - GTJointsHumanEVa (for ground truth HumanEVa)
                            - GTJointsMoVi (for ground truth MoVi)
                            - Mediapipe (for MediaPipe predictions)
                            If None, defaults to PredJointsDeepLabCut (RTMW)
        """
        self.video_json_folder = Path(video_json_folder)
        self.excel_file = Path(excel_file)
        self.joint_enum = joint_enum_class or PredJointsDeepLabCut

        if not self.video_json_folder.exists():
            raise ValueError(f"Video JSON folder does not exist: {video_json_folder}")

        if not self.excel_file.exists():
            raise ValueError(f"Excel file does not exist: {excel_file}")

        logger.info(f"Initialized with video JSON folder: {self.video_json_folder}")
        logger.info(f"Using Excel file: {self.excel_file}")
        logger.info(f"Using joint enum: {self.joint_enum.__name__}")

    @staticmethod
    def get_joint_index_from_enum(joint_enum: Enum, joint_name: str) -> Optional[int]:
        """
        Get joint index from enum given joint name.

        Args:
            joint_enum: Joint enum class
            joint_name: Joint name (e.g., 'LEFT_HIP')

        Returns:
            Joint index if found, else None
        """
        try:
            joint_member = joint_enum[joint_name]
            index = joint_member.value
            # Handle tuples (for multi-point joints)
            if isinstance(index, tuple):
                return index[0]  # Use first index
            return int(index)
        except KeyError:
            return None

    def load_excel_frequencies(self) -> Dict[str, float]:
        """
        Load best frequencies from Excel file.

        Expected columns: Joint, Best_Frequency (or similar)

        Returns:
            Dictionary mapping joint name to best frequency (Hz)
        """
        try:
            # Read all sheets
            xl_file = pd.ExcelFile(self.excel_file)
            logger.info(f"Available sheets: {xl_file.sheet_names}")

            freq_dict = {}

            # Look for sheet containing "best" and "frequency"
            target_sheet = None
            for sheet in xl_file.sheet_names:
                if "best" in sheet.lower() and "frequency" in sheet.lower():
                    target_sheet = sheet
                    break

            # If no exact match, try last sheet
            if not target_sheet and xl_file.sheet_names:
                target_sheet = xl_file.sheet_names[-1]
                logger.info(f"No 'best_frequency' sheet found, using: {target_sheet}")

            if not target_sheet:
                logger.warning("Could not find suitable sheet in Excel file")
                return {}

            df = pd.read_excel(self.excel_file, sheet_name=target_sheet)
            logger.info(
                f"Loaded sheet '{target_sheet}' with columns: {df.columns.tolist()}"
            )

            # Find frequency column
            freq_col = None
            for col in df.columns:
                if "frequency" in col.lower():
                    freq_col = col
                    break

            if freq_col is None:
                # Use last column as frequency if no explicit match
                freq_col = df.columns[-1]
                logger.warning(
                    f"No 'frequency' column found, using last column: {freq_col}"
                )

            # Find joint column (first column or contains "joint")
            joint_col = None
            for col in df.columns:
                if "joint" in col.lower() or col == df.columns[0]:
                    joint_col = col
                    break

            if joint_col is None:
                joint_col = df.columns[0]
                logger.warning(
                    f"No 'joint' column found, using first column: {joint_col}"
                )

            # Extract joint -> frequency mapping
            for _, row in df.iterrows():
                joint = str(row[joint_col]).strip().upper()
                try:
                    freq = float(row[freq_col])
                    freq_dict[joint] = freq
                except (ValueError, TypeError):
                    continue

            logger.info(
                f"Loaded frequencies for {len(freq_dict)} joints: {list(freq_dict.keys())}"
            )
            return freq_dict

        except Exception as e:
            logger.error(f"Error loading Excel file: {e}")
            return {}

    def extract_frequency_from_folder(self, folder_path: Path) -> Optional[float]:
        """
        Extract frequency from folder name.

        E.g., 'filter_butterworth_18th_14hz' -> 14.0

        Args:
            folder_path: Path to folder

        Returns:
            Frequency in Hz if found, else None
        """
        match = re.search(r"(\d+)hz", folder_path.name, re.IGNORECASE)
        if match:
            return float(match.group(1))
        return None

    def find_video_json_files(self) -> Dict[float, List[Path]]:
        """
        Find all video JSON files organized by frequency folders.

        Returns:
            Dictionary mapping frequency -> list of JSON file paths
        """
        freq_files = {}

        # Iterate through subfolders
        for subfolder in self.video_json_folder.iterdir():
            if not subfolder.is_dir():
                continue

            frequency = self.extract_frequency_from_folder(subfolder)
            if frequency is None:
                logger.warning(
                    f"Could not extract frequency from folder: {subfolder.name}"
                )
                continue

            # Find JSON files in this frequency folder
            json_files = list(subfolder.glob("*.json"))
            if json_files:
                if frequency not in freq_files:
                    freq_files[frequency] = []
                freq_files[frequency].extend(json_files)
                logger.info(f"Found {len(json_files)} JSON files at {frequency}Hz")

        logger.info(f"Total frequencies found: {sorted(freq_files.keys())}")
        return freq_files

    @staticmethod
    def load_pose_data(json_path: Path) -> Dict:
        """
        Load pose data from JSON file.

        Supports formats:
        - {"frames": [{"keypoints": [[x, y, c], ...]}, ...]}
        - {"keypoints": [[x, y, c], ...]}

        Args:
            json_path: Path to JSON file

        Returns:
            Dictionary containing pose data
        """
        try:
            with open(json_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading {json_path}: {e}")
            raise

    @staticmethod
    def extract_joint_x_trajectory(pose_data: Dict, joint_index: int) -> np.ndarray:
        """
        Extract X-coordinate trajectory for a specific joint.

        Args:
            pose_data: Loaded pose data dictionary
            joint_index: Index of the joint to extract

        Returns:
            Array of X-coordinates for the joint over frames
        """
        trajectory = []

        # Handle different pose data formats
        if "frames" in pose_data:
            frames = pose_data["frames"]
        elif "keypoints" in pose_data:
            frames = pose_data["keypoints"]
        else:
            logger.debug(f"Unexpected format, keys: {pose_data.keys()}")
            return np.array([])

        if not frames:
            return np.array([])

        # Extract X-coordinate for each frame
        for frame in frames:
            if isinstance(frame, dict) and "keypoints" in frame:
                kp = frame["keypoints"]
            else:
                kp = frame

            if isinstance(kp, list) and len(kp) > joint_index:
                keypoint = kp[joint_index]
                # Extract x coordinate (first element)
                if isinstance(keypoint, (list, tuple)) and len(keypoint) >= 1:
                    x = float(keypoint[0])
                    # Skip invalid coordinates
                    if not np.isnan(x):
                        trajectory.append(x)

        return np.array(trajectory)

    @staticmethod
    def compute_temporal_jitter_x(x_trajectory: np.ndarray) -> float:
        """
        Compute temporal jitter (standard deviation of frame-to-frame changes) for X-coordinate.

        For X trajectory: x(1), x(2), x(3), ...
        Compute frame-to-frame changes: Δx = x(t) - x(t-1)
        Return: std(Δx)

        Small std → smooth motion (low jitter)
        Large std → jittery motion (high jitter)

        Args:
            x_trajectory: Array of X-coordinates over frames

        Returns:
            Standard deviation of frame-to-frame differences (jitter metric)
        """
        if len(x_trajectory) < 2:
            return 0.0

        # Compute frame-to-frame differences: Δx(t) = x(t) - x(t-1)
        differences = np.diff(x_trajectory, axis=0)

        # Return standard deviation of differences
        jitter = float(np.std(differences))
        return jitter

    def process_single_video_json(
        self, json_path: Path, joint_indices: Dict[str, int]
    ) -> Dict[str, float]:
        """
        Process a single video JSON file and compute jitter for specified joints (X-coordinate only).

        Args:
            json_path: Path to JSON file
            joint_indices: Dictionary mapping joint name to joint index

        Returns:
            Dictionary mapping joint name to jitter value
        """
        try:
            pose_data = self.load_pose_data(json_path)
            jitter_per_joint = {}

            for joint_name, joint_idx in joint_indices.items():
                x_trajectory = self.extract_joint_x_trajectory(pose_data, joint_idx)

                if len(x_trajectory) > 1:
                    jitter = self.compute_temporal_jitter_x(x_trajectory)
                    jitter_per_joint[joint_name] = jitter
                else:
                    jitter_per_joint[joint_name] = 0.0

            return jitter_per_joint

        except Exception as e:
            logger.error(f"Error processing {json_path}: {e}")
            return {}

    def run_full_pipeline(
        self, output_excel: Optional[str] = None
    ) -> Tuple[Dict, Dict, Dict]:
        """
        Run the complete temporal stability analysis pipeline.

        Args:
            output_excel: Optional path to save detailed Excel report

        Returns:
            Tuple of (results_dict, statistics_dict, best_frequencies_dict)
        """
        logger.info("=" * 70)
        logger.info("Starting Temporal Stability Analysis Pipeline")
        logger.info("=" * 70)

        # Step 1: Load best frequencies per joint from Excel
        best_freq_dict = self.load_excel_frequencies()
        if not best_freq_dict:
            logger.error("Failed to load best frequencies from Excel")
            return {}, {}, {}

        # Step 2: Build joint index mapping from enum
        joint_indices = {}
        for joint_name, target_freq in best_freq_dict.items():
            joint_idx = self.get_joint_index_from_enum(self.joint_enum, joint_name)
            if joint_idx is not None:
                joint_indices[joint_name] = joint_idx
                logger.info(f"Mapped {joint_name} -> index {joint_idx}")
            else:
                logger.warning(f"Could not map joint {joint_name} from enum")

        if not joint_indices:
            logger.error("Could not map any joints from enum")
            return {}, {}, best_freq_dict

        # Step 3: Find video JSON files organized by frequency
        freq_to_files = self.find_video_json_files()
        if not freq_to_files:
            logger.error("No video JSON files found organized by frequency")
            return {}, {}, best_freq_dict

        # Step 4: Process files at best frequencies
        results = {}  # {joint_name: {video_file: jitter}}
        for joint_name, best_freq in best_freq_dict.items():
            results[joint_name] = {}

            if best_freq not in freq_to_files:
                logger.warning(
                    f"No JSON files found for best frequency {best_freq}Hz of {joint_name}"
                )
                continue

            logger.info(f"\nProcessing {joint_name} at {best_freq}Hz")
            json_files = freq_to_files[best_freq]

            for json_path in json_files:
                logger.info(f"  Processing: {json_path.name}")

                jitter_dict = self.process_single_video_json(
                    json_path, {joint_name: joint_indices[joint_name]}
                )

                if joint_name in jitter_dict:
                    results[joint_name][json_path.stem] = jitter_dict[joint_name]

        # Step 5: Compute statistics per joint
        stats = {}
        for joint_name, video_jitters in results.items():
            if video_jitters:
                jitter_values = list(video_jitters.values())
                stats[joint_name] = {
                    "mean_jitter": np.mean(jitter_values),
                    "std_jitter": np.std(jitter_values),
                    "min_jitter": np.min(jitter_values),
                    "max_jitter": np.max(jitter_values),
                    "num_videos": len(jitter_values),
                    "frequency": best_freq_dict[joint_name],
                }

        # Step 6: Generate Excel report
        logger.info("\nGenerating Excel report...")
        if not output_excel:
            output_excel = self.video_json_folder / "temporal_stability_report.xlsx"

        self.save_detailed_excel_report(results, stats, output_excel)

        logger.info("=" * 70)
        logger.info("Pipeline completed successfully!")
        logger.info("=" * 70)

        return results, stats, best_freq_dict

    def save_detailed_excel_report(self, results: Dict, stats: Dict, output_file: str):
        """
        Save comprehensive Excel report with multiple sheets.

        Args:
            results: Results from pipeline (joint -> video -> jitter)
            stats: Statistics per joint
            output_file: Path to save Excel file
        """
        wb = Workbook()
        wb.remove(wb.active)

        # Sheet 1: Summary Statistics
        ws_summary = wb.create_sheet("Summary_Statistics", 0)
        ws_summary.append(
            [
                "Joint",
                "Best_Frequency_Hz",
                "Mean_Jitter",
                "Std_Jitter",
                "Min_Jitter",
                "Max_Jitter",
                "Num_Videos",
            ]
        )

        for joint_name in sorted(stats.keys()):
            stat_dict = stats[joint_name]
            ws_summary.append(
                [
                    joint_name,
                    stat_dict["frequency"],
                    round(stat_dict["mean_jitter"], 8),
                    round(stat_dict["std_jitter"], 8),
                    round(stat_dict["min_jitter"], 8),
                    round(stat_dict["max_jitter"], 8),
                    stat_dict["num_videos"],
                ]
            )

        self._apply_excel_formatting(ws_summary)

        # Sheet 2: Detailed Results per Video
        ws_detailed = wb.create_sheet("Detailed_Results", 1)
        ws_detailed.append(["Joint", "Video_File", "Jitter_X", "Frequency_Hz"])

        for joint_name in sorted(results.keys()):
            freq = stats[joint_name]["frequency"] if joint_name in stats else "N/A"
            for video_file in sorted(results[joint_name].keys()):
                jitter = results[joint_name][video_file]
                ws_detailed.append([joint_name, video_file, round(jitter, 8), freq])

        self._apply_excel_formatting(ws_detailed)

        # Sheet 3: Interpretation Guide
        ws_guide = wb.create_sheet("Interpretation", 2)
        ws_guide.append(["Metric", "Description"])
        ws_guide.append(
            ["Mean_Jitter", "Average temporal jitter (std of frame-to-frame X changes)"]
        )
        ws_guide.append(["Std_Jitter", "Standard deviation of jitter across videos"])
        ws_guide.append(["Min_Jitter", "Minimum jitter value among videos"])
        ws_guide.append(["Max_Jitter", "Maximum jitter value among videos"])
        ws_guide.append(
            [
                "Jitter_Interpretation",
                "Low jitter = smooth motion; High jitter = noisy motion",
            ]
        )
        ws_guide.append(
            [
                "X_Coordinate_Only",
                "Only X-position is analyzed, Y-coordinate is ignored",
            ]
        )

        self._apply_excel_formatting(ws_guide)

        wb.save(output_file)
        logger.info(f"Detailed Excel report saved to: {output_file}")

    @staticmethod
    def _apply_excel_formatting(worksheet):
        """Apply basic formatting to Excel worksheet."""
        header_fill = PatternFill(
            start_color="4472C4", end_color="4472C4", fill_type="solid"
        )
        header_font = Font(bold=True, color="FFFFFF")
        border = Border(
            left=Side(style="thin"),
            right=Side(style="thin"),
            top=Side(style="thin"),
            bottom=Side(style="thin"),
        )

        for cell in worksheet[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.border = border
            cell.alignment = Alignment(
                horizontal="center", vertical="center", wrap_text=True
            )

        for row in worksheet.iter_rows(min_row=2):
            for cell in row:
                cell.border = border
                cell.alignment = Alignment(horizontal="center", vertical="center")

        # Auto-adjust column widths
        for column in worksheet.columns:
            max_length = 0
            column_letter = get_column_letter(column[0].column)
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 50)
            worksheet.column_dimensions[column_letter].width = adjusted_width


def main():
    """
    Main entry point for the temporal stability calculator.
    Arguments are passed directly here instead of via command line.
    """

    # ============================
    # USER-DEFINED ARGUMENTS
    # ============================

    # Path to folder containing frequency-wise video JSON folders
    video_json_folder = "./pose_videos"

    # Excel file containing best frequency per joint (from PCK analysis)
    best_freq_excel = "./best_frequencies.xlsx"

    # Output Excel report path
    output_excel = "./temporal_stability_report.xlsx"

    # Joint enum to use
    # Options:
    #   PredJointsDeepLabCut
    #   GTJointsHumanSC3D
    #   GTJointsHumanEVa
    #   GTJointsMoVi
    #   Mediapipe
    joint_enum = PredJointsDeepLabCut

    # ============================
    # RUN PIPELINE
    # ============================

    logger.info(f"Using joint enum: {joint_enum.__name__}")
    logger.info(f"Video JSON folder: {video_json_folder}")
    logger.info(f"Best frequency Excel: {best_freq_excel}")
    logger.info(f"Output Excel: {output_excel}")

    try:
        calculator = TemporalStabilityCalculator(
            video_json_folder=video_json_folder,
            excel_file=best_freq_excel,
            joint_enum_class=joint_enum,
        )

        results, stats, best_freq_dict = calculator.run_full_pipeline(
            output_excel=output_excel
        )

        # ============================
        # PRINT SUMMARY TO CONSOLE
        # ============================

        print("\n" + "=" * 70)
        print("TEMPORAL STABILITY SUMMARY (X-Coordinate Only)")
        print("=" * 70)

        for joint_name in sorted(stats.keys()):
            stat = stats[joint_name]
            print(
                f"{joint_name:20} @ {stat['frequency']:6.1f} Hz | "
                f"Mean Jitter = {stat['mean_jitter']:.8f} "
                f"(± {stat['std_jitter']:.8f}) | "
                f"Videos = {stat['num_videos']}"
            )

        print("=" * 70)

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()

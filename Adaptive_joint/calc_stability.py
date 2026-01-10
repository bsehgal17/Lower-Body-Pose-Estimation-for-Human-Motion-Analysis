"""
Temporal Stability Calculator for Video Pose Data using Enum

This script:
1. Reads an Excel file to extract the best frequency for each joint (highest PCK)
2. Processes video JSON files organized by frequency folders
3. Calculates temporal stability (jitter) of X-coordinates for each joint
4. Uses joint enum to map joint names to indices
5. Generates comprehensive Excel report with stability metrics
"""

import json
import re
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from openpyxl import Workbook
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from enum import Enum

# Example Enum for DeepLabCut joints


class PredJointsDeepLabCut(Enum):
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16


# Configure logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class TemporalStabilityCalculator:
    """Calculate temporal stability (jitter) for X-coordinate of joint trajectories from video JSON files."""

    def __init__(self, video_json_folder: str, excel_file: str, joint_enum_class: Enum = PredJointsDeepLabCut):
        self.video_json_folder = Path(video_json_folder)
        self.excel_file = Path(excel_file)
        self.joint_enum = joint_enum_class

        if not self.video_json_folder.exists():
            raise ValueError(
                f"Video JSON folder does not exist: {video_json_folder}")
        if not self.excel_file.exists():
            raise ValueError(f"Excel file does not exist: {excel_file}")

        logger.info(
            f"Initialized with video JSON folder: {self.video_json_folder}")
        logger.info(f"Using Excel file: {self.excel_file}")
        logger.info(f"Using joint enum: {self.joint_enum.__name__}")

    @staticmethod
    def get_joint_index_from_enum(joint_enum: Enum, joint_name: str) -> Optional[int]:
        """Map a joint name (string) to its Enum index."""
        try:
            value = joint_enum[joint_name.upper()].value
            if isinstance(value, tuple):
                return value[0]
            return int(value)
        except KeyError:
            return None

    def load_excel_frequencies(self) -> Dict[str, Dict[str, list]]:
        """Load best frequencies per joint per video from Excel."""
        try:
            xl_file = pd.ExcelFile(self.excel_file)
            target_sheet = None
            for sheet in xl_file.sheet_names:
                if "best" in sheet.lower() and "frequency" in sheet.lower():
                    target_sheet = sheet
                    break
            if not target_sheet:
                target_sheet = xl_file.sheet_names[-1]

            df = pd.read_excel(self.excel_file, sheet_name=target_sheet)
            video_col = df.columns[0]
            joint_cols = df.columns[1:]

            freq_dict = {}
            for _, row in df.iterrows():
                video_name = str(row[video_col])
                freq_dict[video_name] = {}
                for joint in joint_cols:
                    cell_value = row[joint]
                    if pd.isna(cell_value):
                        freq_list = []
                    else:
                        freq_list = [int(f.strip())
                                     for f in str(cell_value).split(',')]
                    freq_dict[video_name][joint.upper()] = freq_list

            logger.info(
                f"Loaded frequencies for {len(freq_dict)} videos and {len(joint_cols)} joints from sheet '{target_sheet}'.")
            return freq_dict

        except Exception as e:
            logger.error(f"Error loading Excel file: {e}")
            return {}

    @staticmethod
    def _apply_excel_formatting(ws):
        """Apply basic formatting to an Excel worksheet."""
        header_fill = PatternFill(
            start_color="4472C4", end_color="4472C4", fill_type="solid")
        header_font = Font(bold=True, color="FFFFFF")
        border = Border(left=Side(style="thin"), right=Side(style="thin"),
                        top=Side(style="thin"), bottom=Side(style="thin"))

        # Format header row
        for cell in ws[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.border = border
            cell.alignment = Alignment(horizontal="center", vertical="center")

        # Format all other cells
        for row in ws.iter_rows(min_row=2):
            for cell in row:
                cell.border = border
                cell.alignment = Alignment(
                    horizontal="center", vertical="center")

        # Adjust column widths
        for column in ws.columns:
            max_length = max(
                len(str(cell.value)) if cell.value is not None else 0 for cell in column)
            ws.column_dimensions[get_column_letter(
                column[0].column)].width = min(max_length + 2, 50)

    @staticmethod
    def extract_frequency_from_folder(folder_path: Path) -> Optional[float]:
        match = re.search(r"(\d+)hz", folder_path.name, re.IGNORECASE)
        return float(match.group(1)) if match else None

    def find_video_json_files(self) -> Dict[float, List[Path]]:
        """Find JSON files organized by frequency folders (recursive)."""
        freq_files = {}
        for subfolder in self.video_json_folder.iterdir():
            if not subfolder.is_dir():
                continue
            match = re.search(r"(\d+(\.\d+)?)\s*hz",
                              subfolder.name, re.IGNORECASE)
            if not match:
                continue
            freq = float(match.group(1))
            json_files = list(subfolder.rglob("*.json"))  # recursive search
            if json_files:
                freq_files[freq] = json_files
        if not freq_files:
            logger.warning(f"No JSON files found in {self.video_json_folder}")
        return freq_files

    @staticmethod
    def parse_video_name(video_name: str) -> Dict[str, str]:
        """
        Extract subject, motion, camera from Excel video name like 'S1_Box_1_1.0'
        Returns camera as integer for path matching.
        """
        pattern = r"(S\d+)_(\w+_\d+)_(\d+(?:\.\d+)?)"  # integer or float
        match = re.match(pattern, video_name)
        if not match:
            return {}
        subject, motion, cam_number = match.groups()
        # Convert float string like "1.0" → int 1
        cam_int = int(float(cam_number))
        return {"subject": subject, "motion": motion, "camera": cam_int}

    @staticmethod
    def parse_json_path(json_path: Path) -> Dict[str, str]:
        """Extract subject, motion, camera from JSON path."""
        parts = json_path.parts
        subject_match = next((p for p in parts if re.match(r"S\d+", p)), None)
        video_folder = json_path.stem
        motion_match = re.match(r"(\w+_\d+)(?:_?\(?C(\d+)\)?)?", video_folder)
        motion = motion_match.group(1) if motion_match else None
        camera = motion_match.group(2) if motion_match else None
        camera = int(camera)
        return {"subject": subject_match, "motion": motion, "camera": camera}

    @staticmethod
    def load_pose_data(json_path: Path) -> Dict:
        with open(json_path, "r") as f:
            return json.load(f)

    @staticmethod
    def extract_joint_x_trajectory(pose_data: dict, joint_index: int) -> np.ndarray:
        """Extract X-coordinate trajectory of a joint across all frames and persons."""
        trajectory = []
        persons = pose_data.get("persons", [])
        if not persons:
            return np.array(trajectory)

        for person in persons:
            poses = person.get("poses", [])
            for frame in poses:
                keypoints = frame.get("keypoints", [])
                if len(keypoints) == 1 and isinstance(keypoints[0], list):
                    keypoints = keypoints[0]
                if isinstance(keypoints, list) and len(keypoints) > joint_index:
                    joint_kp = keypoints[joint_index]
                    if isinstance(joint_kp, (list, tuple)) and len(joint_kp) >= 1:
                        x = joint_kp[0]
                        if x is not None and not np.isnan(x):
                            trajectory.append(x)
        return np.array(trajectory)

    @staticmethod
    def compute_temporal_jitter_x(x_trajectory: np.ndarray) -> float:
        if len(x_trajectory) < 2:
            return 0.0
        return float(np.std(np.diff(x_trajectory, axis=0)))

    def process_single_video_json(self, json_path: Path, joint_indices: Dict[str, int]) -> Dict[str, float]:
        pose_data = self.load_pose_data(json_path)
        return {joint_name: self.compute_temporal_jitter_x(
            self.extract_joint_x_trajectory(pose_data, joint_idx))
            for joint_name, joint_idx in joint_indices.items()}

    def run_full_pipeline(self, output_excel: Optional[str] = None) -> Tuple[Dict, Dict, Dict, Dict]:
        """Run full pipeline, return results, original frequencies, all jitters."""
        best_freq_dict = self.load_excel_frequencies()
        joint_names_set = set()
        for video_dict in best_freq_dict.values():
            joint_names_set.update(video_dict.keys())

        joint_indices = {
            name: TemporalStabilityCalculator.get_joint_index_from_enum(
                self.joint_enum, name)
            for name in joint_names_set
            if TemporalStabilityCalculator.get_joint_index_from_enum(self.joint_enum, name) is not None
        }

        freq_to_files = self.find_video_json_files()
        results = {}      # Best jitter per joint/video
        all_jitters = {}  # All jitters per freq per joint/video

        for video_name, joints_dict in best_freq_dict.items():
            video_info = self.parse_video_name(video_name)
            if not video_info:
                continue

            for joint_name, candidate_freqs in joints_dict.items():
                joint_idx = joint_indices.get(joint_name)
                if joint_idx is None or not candidate_freqs:
                    continue

                best_jitter = None
                best_freq_for_joint = None
                if joint_name not in all_jitters:
                    all_jitters[joint_name] = {}
                all_jitters[joint_name][video_name] = {}

                for freq in candidate_freqs:
                    json_files = freq_to_files.get(freq, [])

                    # Filter JSON files to match subject/motion/camera
                    matching_jsons = []
                    for jpath in json_files:
                        json_info = self.parse_json_path(jpath)
                        if not json_info:
                            continue
                        if (video_info["subject"] == json_info["subject"]
                                and video_info["motion"].lower() == json_info["motion"].lower()
                                and video_info['camera'] == json_info["camera"]):
                            matching_jsons.append(jpath)

                    if not matching_jsons:
                        logger.warning(
                            f"No JSON files match video '{video_name}' at freq {freq}Hz")
                        continue

                    # Compute average jitter across matching JSON files
                    jitters = []
                    for json_path in matching_jsons:
                        jitter_dict = self.process_single_video_json(
                            json_path, {joint_name: joint_idx})
                        jitters.append(jitter_dict.get(joint_name, 0.0))
                    mean_jitter = np.mean(jitters) if jitters else None

                    # Save all jitters per freq
                    all_jitters[joint_name][video_name][freq] = mean_jitter

                    # Update best freq if this mean_jitter is lower
                    if best_jitter is None or (mean_jitter is not None and mean_jitter < best_jitter):
                        best_jitter = mean_jitter
                        best_freq_for_joint = freq

                # Store best freq in results
                if best_jitter is not None and best_freq_for_joint is not None:
                    if joint_name not in results:
                        results[joint_name] = {}
                    results[joint_name][video_name] = {
                        "best_freq": best_freq_for_joint,
                        "jitter": best_jitter
                    }

        # Save Excel if requested
        if output_excel:
            self.save_detailed_excel_report(results, all_jitters, output_excel)

        return results, best_freq_dict, all_jitters

    def save_detailed_excel_report(self, results: Dict, all_jitters: Dict, output_file: str):
        """
        Sheet 1: All_Jitters
            Joint | Video | Frequency | Jitter
        Sheet 2: Best_Frequency
            Rows: Video names
            Columns: Joint names
            Values: Best frequency (Hz)
        """
        wb = Workbook()
        wb.remove(wb.active)

        # -------------------------
        # Sheet 1: All jitters
        # -------------------------
        ws_all = wb.create_sheet("All_Jitters")
        ws_all.append(["Joint", "Video_Name", "Frequency_Hz", "Jitter_X"])

        for joint, videos in all_jitters.items():
            for video, freq_dict in videos.items():
                for freq, jitter in freq_dict.items():
                    ws_all.append([
                        joint,
                        video,
                        freq,
                        round(jitter, 8) if jitter is not None else None
                    ])

        self._apply_excel_formatting(ws_all)

        # -------------------------
        # Sheet 2: Best frequency (matrix)
        # -------------------------
        ws_best = wb.create_sheet("Best_Frequency")

        # Collect sorted videos and joints
        videos = sorted(
            {video for joint_data in results.values()
             for video in joint_data.keys()}
        )
        joints = sorted(results.keys())

        # Header row
        ws_best.append(["Video_Name"] + joints)

        # Fill table
        for video in videos:
            row = [video]
            for joint in joints:
                if joint in results and video in results[joint]:
                    row.append(results[joint][video]["best_freq"])
                else:
                    row.append(None)
            ws_best.append(row)

        self._apply_excel_formatting(ws_best)

        wb.save(output_file)
        logger.info(f"Excel report saved to: {output_file}")


def main():
    video_json_folder = r"/storageh100/Projects/Gaitly/bsehgal/pipeline_results/HumanEva/Butterworth_filter"
    best_freq_excel = r"/storage/Projects/Gaitly/bsehgal/lower_body_pose_est/pipeline_results/Adaptive_filt/pck_summary.xlsx"
    output_excel = r"/storage/Projects/Gaitly/bsehgal/lower_body_pose_est/pipeline_results/Adaptive_filt/temporal_stability_report.xlsx"
    joint_enum = PredJointsDeepLabCut

    calculator = TemporalStabilityCalculator(
        video_json_folder=video_json_folder,
        excel_file=best_freq_excel,
        joint_enum_class=joint_enum
    )

    results, best_freq_dict, all_jitters = calculator.run_full_pipeline(
        output_excel)


if __name__ == "__main__":
    main()

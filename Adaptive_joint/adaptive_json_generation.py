"""
Adaptive Joint Filtering JSON Generation

Workflow:
1. Read Excel file with Best Frequency sheet
2. For each video row, get frequency from joint column
3. Extract frequency from filter folder path (e.g., filter_butterworth_18th_14hz -> 14hz)
4. Load filtered data for that video from the corresponding frequency folder
5. Generate JSON files with adaptive filtered data (one best frequency per joint per video)
6. Save in "adaptive_filtering" folder maintaining original path structure
"""

import os
import json
import re
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from copy import deepcopy

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class AdaptiveJSONGenerator:
    """Generate adaptive filtered JSON files based on best frequencies per joint."""

    def __init__(
        self,
        excel_path: str,
        filter_base_path: str,
        output_base_path: str = "adaptive_filtering",
    ):
        """
        Initialize the adaptive JSON generator.

        Args:
            excel_path: Path to Excel file with "Best Frequency" sheet
            filter_base_path: Path to filter results folder
                             e.g., /storageh100/Projects/Gaitly/bsehgal/pipeline_results/HumanEva/Butterworth_filter/
            output_base_path: Base output folder for adaptive JSON files
        """
        self.excel_path = Path(excel_path)
        self.filter_base_path = Path(filter_base_path)
        self.output_base_path = Path(output_base_path)
        self.excel_data = None
        self.joint_freq_map = {}  # {video_name: {joint_name: frequency}}
        self.filter_folder_map = {}  # {frequency: path_to_filter_folder}

        logger.info(f"Initialized AdaptiveJSONGenerator")
        logger.info(f"  Excel Path: {self.excel_path}")
        logger.info(f"  Filter Base Path: {self.filter_base_path}")
        logger.info(f"  Output Base Path: {self.output_base_path}")

    def extract_frequency_from_path(self, path: str) -> Optional[float]:
        """
        Extract filter frequency from folder name.
        E.g., filter_butterworth_18th_14hz -> 14
        """
        match = re.search(r"(\d+)hz", path, re.IGNORECASE)
        if match:
            return float(match.group(1))
        return None

    def load_excel_best_frequencies(self) -> bool:
        """
        Load Excel file and extract best frequency for each joint per video.

        Expected structure:
        - Sheet name: "Best Frequency"
        - First column: Video name (e.g., "S1_Box_1_1.0")
        - Remaining columns: Joint names with their best frequencies
        """
        try:
            if not self.excel_path.exists():
                logger.error(f"Excel file not found: {self.excel_path}")
                return False

            # Read the "Best Frequency" sheet
            df = pd.read_excel(self.excel_path, sheet_name="Best Frequency")
            logger.info(f"Loaded Excel file with {len(df)} rows")
            logger.info(f"Columns: {df.columns.tolist()}")

            # Parse the Excel data into a structured format
            # First column is video name, rest are joints with frequencies
            first_col_name = df.columns[0]

            for idx, row in df.iterrows():
                video_name = str(row[first_col_name]).strip()

                if video_name in ["", "NaN", None] or pd.isna(row[first_col_name]):
                    logger.debug(f"Skipping row {idx}: empty video name")
                    continue

                logger.info(f"Processing video: {video_name}")

                # For each column after the first, assume it's a joint with frequency value
                for col_idx, col_name in enumerate(df.columns[1:], start=1):
                    joint_name = str(col_name).strip()
                    freq_value = row.iloc[col_idx]

                    if pd.notna(freq_value) and freq_value != "":
                        try:
                            freq = float(freq_value)
                            if video_name not in self.joint_freq_map:
                                self.joint_freq_map[video_name] = {}
                            self.joint_freq_map[video_name][joint_name] = freq
                            logger.debug(f"  {joint_name}: {freq}Hz")
                        except (ValueError, TypeError):
                            logger.warning(
                                f"Could not convert frequency for {video_name}/{joint_name}: {freq_value}"
                            )

            logger.info(
                f"\nLoaded frequency data for {len(self.joint_freq_map)} videos"
            )
            for video_name, joints in list(self.joint_freq_map.items())[
                :5
            ]:  # Show first 5
                logger.info(f"  {video_name}: {len(joints)} joints")

            return True

        except Exception as e:
            logger.error(f"Error loading Excel file: {e}")
            import traceback

            traceback.print_exc()
            return False

    def build_filter_folder_map(self) -> bool:
        """
        Build a mapping of frequencies to their corresponding filter folders.

        Scans the filter_base_path to find all frequency-specific folders.
        """
        try:
            if not self.filter_base_path.exists():
                logger.error(
                    f"Filter base path not found: {self.filter_base_path}")
                return False

            # Find all frequency folders
            for item in self.filter_base_path.iterdir():
                if item.is_dir():
                    freq = self.extract_frequency_from_path(item.name)
                    if freq is not None:
                        self.filter_folder_map[freq] = item
                        logger.debug(
                            f"Found frequency {freq}Hz folder: {item.name}")

            logger.info(
                f"Built map for {len(self.filter_folder_map)} frequencies")
            logger.info(
                f"Available frequencies: {sorted(self.filter_folder_map.keys())}Hz"
            )
            return True

        except Exception as e:
            logger.error(f"Error building filter folder map: {e}")
            return False

    def find_filtered_json_files(
        self, video_name: str, frequency: float
    ) -> Optional[List[Path]]:
        """
        Find JSON files for a specific video and frequency in the filter folders.

        Path structure:
        /filter/filter_butterworth_18th_1hz/filter/2025-12-16_02-26-06/butterworth_order18_cutoff1.0_fs60.0/S1/Image_Data/Box_1_(C1)/Box_1_(C1).json

        Args:
            video_name: Name of video (e.g., "S1_Box_1_1.0" or "Box_1_(C1)")
            frequency: Filter frequency (e.g., 1.0)

        Returns:
            List of JSON file paths matching the video name, or None
        """
        if frequency not in self.filter_folder_map:
            logger.warning(
                f"No filter folder found for frequency {frequency}Hz")
            return None

        freq_folder = self.filter_folder_map[frequency]

        # Navigate to filter/ subdirectory
        filter_subdir = freq_folder / "filter"
        if not filter_subdir.exists():
            logger.warning(f"No 'filter' subdirectory found in {freq_folder}")
            return None

        # Search through all timestamp directories (2025-12-16_02-26-06, etc.)
        matching_files = []

        try:
            for timestamp_dir in filter_subdir.iterdir():
                if not timestamp_dir.is_dir():
                    continue

                logger.debug(
                    f"Searching in timestamp directory: {timestamp_dir.name}")

                # Search through all filter parameter folders
                for param_dir in timestamp_dir.iterdir():
                    if not param_dir.is_dir():
                        continue

                    # Now search recursively for JSON files matching the video name
                    for json_file in param_dir.rglob("*.json"):
                        # Check if video name matches filename or any parent directory
                        file_stem = json_file.stem.lower()
                        video_name_lower = video_name.lower()

                        # Extract core name (remove special characters for comparison)
                        file_core = re.sub(r"[_()\-\s]", "", file_stem)
                        video_core = re.sub(r"[_()\-\s]", "", video_name_lower)

                        # Check multiple matching strategies
                        if (
                            video_name_lower in file_stem
                            or file_stem == video_name_lower
                            or file_core == video_core
                            or json_file.parent.name.lower() == video_name_lower
                            or json_file.parent.name.lower() in video_name_lower
                        ):
                            matching_files.append(json_file)
                            logger.debug(f"Found filtered file: {json_file}")
                            break  # Found the file for this video, move to next timestamp

                # If we found the file, no need to search other timestamps
                if matching_files:
                    break

        except Exception as e:
            logger.warning(f"Error searching filter directories: {e}")
            return None

        if not matching_files:
            logger.warning(
                f"No filtered JSON files found for '{video_name}' at {frequency}Hz in {filter_subdir}"
            )
            logger.debug(
                f"Searched pattern in directories under: {filter_subdir}")
            return None

        return matching_files

    def load_filtered_data(self, json_path: Path) -> Optional[Dict]:
        """
        Load filtered data from JSON file.

        Supports both new format (with persons) and legacy format.
        """
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
            logger.debug(f"Loaded filtered data from: {json_path}")
            return data
        except Exception as e:
            logger.error(f"Error loading JSON from {json_path}: {e}")
            return None

    def extract_joint_data_from_poses(
        self, poses: List[Dict], joint_idx: int
    ) -> List[List[float]]:
        """
        Extract specific joint coordinates from poses list.

        Args:
            poses: List of pose dictionaries with keypoints
            joint_idx: Index of the joint to extract

        Returns:
            List of [x, y] coordinates for the joint across all frames
        """
        joint_coords = []
        for pose in poses:
            keypoints = pose.get("keypoints", [])
            if joint_idx < len(keypoints):
                joint_coords.append(keypoints[joint_idx])
            else:
                logger.warning(f"Joint index {joint_idx} out of range")
                joint_coords.append([0, 0])
        return joint_coords

    def generate_adaptive_json(
        self,
        base_video_data: Dict,
        joint_filtered_data: Dict[int, Dict],
        video_name: str,
    ) -> Dict:
        """
        Generate adaptive JSON by replacing joint data with best filtered versions.

        Args:
            base_video_data: Original video data structure
            joint_filtered_data: {joint_idx: filtered_data_dict}
            video_name: Name of the video

        Returns:
            Modified video data with adaptive filtered keypoints
        """
        adaptive_data = deepcopy(base_video_data)

        # Handle new format with persons
        if "persons" in adaptive_data:
            for person in adaptive_data["persons"]:
                if "poses" in person:
                    poses = person["poses"]

                    # For each pose, update keypoints with adaptive filtered data
                    for pose_idx, pose in enumerate(poses):
                        frame_idx = pose.get("frame_idx", pose_idx)
                        keypoints = pose.get("keypoints", [])

                        # Update each joint that has adaptive filtered data
                        for joint_idx, filtered_source in joint_filtered_data.items():
                            if (
                                "persons" in filtered_source
                                and len(filtered_source["persons"]) > 0
                            ):
                                filtered_person = filtered_source["persons"][0]
                                if "poses" in filtered_person:
                                    filtered_poses = filtered_person["poses"]
                                    # Find pose with matching frame_idx
                                    for fpose in filtered_poses:
                                        if fpose.get("frame_idx", -1) == frame_idx:
                                            fkpts = fpose.get("keypoints", [])
                                            if joint_idx < len(
                                                fkpts
                                            ) and joint_idx < len(keypoints):
                                                keypoints[joint_idx] = fkpts[joint_idx]
                                            break

                        pose["keypoints"] = keypoints

        # Handle legacy format with keypoints as list
        elif "keypoints" in adaptive_data:
            frames = adaptive_data["keypoints"]
            for frame_idx, frame_data in enumerate(frames):
                if "keypoints" in frame_data:
                    for person_idx, person_kpts in enumerate(frame_data["keypoints"]):
                        # Update each joint that has adaptive filtered data
                        for joint_idx, filtered_source in joint_filtered_data.items():
                            if "keypoints" in filtered_source and frame_idx < len(
                                filtered_source["keypoints"]
                            ):
                                fframe = filtered_source["keypoints"][frame_idx]
                                if "keypoints" in fframe and person_idx < len(
                                    fframe["keypoints"]
                                ):
                                    fkpts = fframe["keypoints"][person_idx]
                                    if joint_idx < len(fkpts) and joint_idx < len(
                                        person_kpts
                                    ):
                                        person_kpts[joint_idx] = fkpts[joint_idx]

        return adaptive_data

    def get_joint_enum_mapping(self) -> Dict[str, int]:
        """
        Get joint name to index mapping.

        Should match the keypoint format used in the dataset.
        Common formats: COCO-17, COCO-133, H36M-17, etc.
        """
        # Default mapping for lower-body (H36M-like)
        joint_mapping = {
            "pelvis": 0,
            "left_hip": 1,
            "left_knee": 2,
            "left_ankle": 3,
            "right_hip": 4,
            "right_knee": 5,
            "right_ankle": 6,
            "spine": 7,
            "thorax": 8,
            "upper_neck": 9,
            "head": 10,
            "left_shoulder": 11,
            "left_elbow": 12,
            "left_wrist": 13,
            "right_shoulder": 14,
            "right_elbow": 15,
            "right_wrist": 16,
            # Upper case variants
            "LEFT_HIP": 1,
            "LEFT_KNEE": 2,
            "LEFT_ANKLE": 3,
            "RIGHT_HIP": 4,
            "RIGHT_KNEE": 5,
            "RIGHT_ANKLE": 6,
            "PELVIS": 0,
            "SPINE": 7,
            "THORAX": 8,
            "UPPER_NECK": 9,
            "HEAD": 10,
            "LEFT_SHOULDER": 11,
            "LEFT_ELBOW": 12,
            "LEFT_WRIST": 13,
            "RIGHT_SHOULDER": 14,
            "RIGHT_ELBOW": 15,
            "RIGHT_WRIST": 16,
        }
        return joint_mapping

    def process_video(self, video_name: str) -> bool:
        """
        Process a single video: generate adaptive JSON with best frequencies.

        Args:
            video_name: Name of the video to process (e.g., "S1_Box_1_1.0")

        Returns:
            True if successful, False otherwise
        """
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Processing video: {video_name}")
        logger.info(f"{'=' * 80}")

        if video_name not in self.joint_freq_map:
            logger.warning(f"Video {video_name} not found in frequency map")
            return False

        joint_frequencies = self.joint_freq_map[video_name]
        logger.info(
            f"Found {len(joint_frequencies)} joints with best frequencies")
        for jname, freq in joint_frequencies.items():
            logger.info(f"  {jname}: {freq}Hz")

        # Load filtered data for each unique frequency used by this video
        joint_filtered_data = {}  # {joint_idx: filtered_data}
        joint_enum = self.get_joint_enum_mapping()
        frequencies_used = set(joint_frequencies.values())

        # Load base video data from the lowest frequency (most filtered)
        base_video_data = None
        base_json_path = None

        for frequency in sorted(frequencies_used):
            logger.info(f"\nSearching for filtered data at {frequency}Hz...")
            json_files = self.find_filtered_json_files(video_name, frequency)

            if not json_files:
                logger.warning(
                    f"No filtered data found for {video_name} at {frequency}Hz"
                )
                continue

            json_file = json_files[0]
            logger.info(f"Loading filtered data from: {json_file}")
            filtered_data = self.load_filtered_data(json_file)

            if filtered_data is None:
                continue

            # Store base data if not already set
            if base_video_data is None:
                base_video_data = deepcopy(filtered_data)
                base_json_path = json_file
                logger.info(f"✓ Set base video data from: {json_file.name}")

            # Extract joints that use this frequency
            joints_for_freq = [
                (jname, jfreq)
                for jname, jfreq in joint_frequencies.items()
                if jfreq == frequency
            ]

            logger.info(
                f"Joints using {frequency}Hz: {[j[0] for j in joints_for_freq]}"
            )

            for joint_name, _ in joints_for_freq:
                # Get joint index from mapping
                joint_idx = joint_enum.get(joint_name, -1)

                # Try different naming conventions
                if joint_idx < 0:
                    joint_idx = joint_enum.get(joint_name.lower(), -1)
                if joint_idx < 0:
                    joint_idx = joint_enum.get(joint_name.upper(), -1)

                if joint_idx < 0:
                    logger.warning(
                        f"Could not find joint index for '{joint_name}'. Available: {list(joint_enum.keys())[:5]}..."
                    )
                    continue

                # Store the filtered data for this joint
                joint_filtered_data[joint_idx] = filtered_data
                logger.debug(
                    f"  ✓ Mapped joint '{joint_name}' (idx={joint_idx}) to {frequency}Hz data"
                )

        if base_video_data is None:
            logger.error(f"Failed to load base video data for {video_name}")
            return False

        logger.info(f"\n✓ Loaded data for {len(joint_filtered_data)} joints")

        # Generate adaptive JSON
        adaptive_data = self.generate_adaptive_json(
            base_video_data, joint_filtered_data, video_name
        )

        # Create output directory maintaining the subject/video structure
        # Example: S1/Image_Data/Box_1_(C1)/Box_1_(C1).json
        if base_json_path:
            # Extract the subject folder and video folders from the path
            # From: .../S1/Image_Data/Box_1_(C1)/Box_1_(C1).json
            parts = base_json_path.parts

            # Find subject folder (starts with 'S' followed by number)
            subject_idx = -1
            for i, part in enumerate(parts):
                if re.match(r"^S\d+$", part, re.IGNORECASE):
                    subject_idx = i
                    break

            if subject_idx >= 0:
                # Build relative path from subject folder onwards
                relative_parts = parts[subject_idx:]
                # Replace the filename with video_name.json
                relative_parts = list(
                    relative_parts[:-1]) + [f"{video_name}.json"]
                output_path = self.output_base_path / Path(*relative_parts)
            else:
                # Fallback: use parent structure if subject not found
                output_path = (
                    self.output_base_path
                    / base_json_path.parent.name
                    / f"{video_name}.json"
                )
        else:
            output_path = self.output_base_path / f"{video_name}.json"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save adaptive JSON
        try:
            with open(output_path, "w") as f:
                json.dump(adaptive_data, f, indent=2)
            logger.info(f"✓ Saved adaptive JSON to: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving adaptive JSON: {e}")
            import traceback

            traceback.print_exc()
            return False

    def process_all_videos(self) -> bool:
        """
        Process all videos from the Excel file.

        Returns:
            True if all successful, False otherwise
        """
        if not self.load_excel_best_frequencies():
            logger.error("Failed to load Excel file")
            return False

        if not self.build_filter_folder_map():
            logger.error("Failed to build filter folder map")
            return False

        success_count = 0
        total_videos = len(self.joint_freq_map)

        for video_name in sorted(self.joint_freq_map.keys()):
            if self.process_video(video_name):
                success_count += 1

        logger.info(f"\n{'=' * 80}")
        logger.info(
            f"Processing complete: {success_count}/{total_videos} videos successfully processed"
        )
        logger.info(f"{'=' * 80}\n")

        return success_count == total_videos


def main():
    """Main execution function."""
    # Configuration - Edit these paths
    EXCEL_PATH = (
        # Excel file with Best Frequency sheet
        r"/storage/Projects/Gaitly/bsehgal/lower_body_pose_est/pipeline_results/Adaptive_filt/pck_summary.xlsx"
    )
    FILTER_BASE_PATH = r"/storageh100/Projects/Gaitly/bsehgal/pipeline_results/HumanEva/Butterworth_filter/"
    OUTPUT_BASE_PATH = r"/storage/Projects/Gaitly/bsehgal/lower_body_pose_est/pipeline_results/Adaptive_filt/adaptive_filtering"

    # ========================================================================

    print("\n" + "=" * 80)
    print("Adaptive Joint Filtering JSON Generator")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Excel Path: {EXCEL_PATH}")
    print(f"  Filter Base Path: {FILTER_BASE_PATH}")
    print(f"  Output Base Path: {OUTPUT_BASE_PATH}")
    print("-" * 80 + "\n")

    # Validate paths
    excel_path = Path(EXCEL_PATH)
    if not excel_path.exists():
        print(f"✗ Error: Excel file not found: {EXCEL_PATH}")
        return False

    filter_base = Path(FILTER_BASE_PATH)
    if not filter_base.exists():
        print(f"✗ Error: Filter base path not found: {FILTER_BASE_PATH}")
        return False

    try:
        generator = AdaptiveJSONGenerator(
            excel_path=EXCEL_PATH,
            filter_base_path=FILTER_BASE_PATH,
            output_base_path=OUTPUT_BASE_PATH,
        )

        success = generator.process_all_videos()

        if success:
            print("✓ All videos processed successfully!")
            print(f"✓ Results saved in: {OUTPUT_BASE_PATH}")
            return True
        else:
            print("✗ Some videos failed to process")
            return False

    except Exception as e:
        print(f"✗ Error during processing: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

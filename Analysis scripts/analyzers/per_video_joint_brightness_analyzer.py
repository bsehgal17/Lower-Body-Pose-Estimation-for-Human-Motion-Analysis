"""
Per-Video Joint Brightness Analyzer

Analyzes brightness values at all joint coordinates per video basis.
Takes all six joints in each video and analyzes brightness on those GT joints.
"""

from utils.joint_enum import GTJointsHumanEVa, GTJointsMoVi
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from core.base_classes import BaseAnalyzer
import os
import cv2
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))


class PerVideoJointBrightnessAnalyzer(BaseAnalyzer):
    def visualize_first_frame_with_joint_circles(
        self,
        video_name: str,
        video_data: pd.DataFrame,
        group_key=None,
        grouping_cols: List[str] = None,
        output_dir: str = None,
        sync_offset: int = 0
    ):
        """Visualize the first frame of the video (after sync offset) with circles around each joint used for brightness sampling."""
        # Load ground truth coordinates for this video (without sync offset)
        gt_coordinates = self._load_video_ground_truth(
            video_name, video_data, group_key, grouping_cols
        )
        if not gt_coordinates:
            print(f"   No ground truth coordinates found for {video_name}")
            return

        # Load video file
        video_path = self._find_video_path(
            video_name, video_data, group_key, grouping_cols
        )
        if not video_path or not os.path.exists(video_path):
            print(f"   Video file not found for {video_name}")
            return

        cap = cv2.VideoCapture(video_path)

        # Skip to sync offset frame (the actual first frame we'll use)
        if sync_offset > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, sync_offset)

        ret, frame = cap.read()
        cap.release()

        if not ret:
            print(f"   Could not read frame {sync_offset} from {video_name}")
            return

        # Draw circles for each joint using the coordinates from the first frame of offset GT
        for joint_name, joint_coords in gt_coordinates.items():
            if len(joint_coords) > 0:  # Use the first frame of the offset coordinates
                # This is frame 0 after sync offset is applied
                x, y = joint_coords[0]
                x = int(x)
                y = int(y)
                cv2.circle(frame, (x, y), self.sampling_radius, (0, 0, 255), 2)
                cv2.putText(
                    frame,
                    joint_name,
                    (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )

        # Save or show the frame
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            out_path = os.path.join(
                output_dir, f"{video_name}_frame_{sync_offset}_joints.png")
            cv2.imwrite(out_path, frame)
            print(f"   Saved visualization: {out_path}")
        else:
            cv2.imshow(f"{video_name} - Frame {sync_offset} Joints", frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    """Analyzer for per-video joint brightness analysis across all joints."""

    def __init__(
        self,
        config,
        joint_names: List[str] = None,
        sampling_radius: int = 3,
        dataset_name: str = "movi",
        visualize_joints: bool = False,
        visualization_output_dir: str = None
    ):
        """Initialize the per-video joint brightness analyzer.

        Args:
            config: Configuration object with paths
            joint_names: List of joint names to analyze (uses default if None)
            sampling_radius: Radius for brightness sampling around joint coordinates
            dataset_name: Name of the dataset ('movi' or 'humaneva')
            visualize_joints: Whether to generate visualizations of joint areas
            visualization_output_dir: Directory to save visualizations
        """
        super().__init__(config)
        self.sampling_radius = sampling_radius
        self.dataset_name = dataset_name.lower()
        self.visualize_joints = visualize_joints
        self.visualization_output_dir = visualization_output_dir

        # Set up joint names
        if joint_names is None:
            self.joint_names = self._get_default_joint_names()
        else:
            self.joint_names = joint_names

        print(
            f"Initialized per-video analysis for {len(self.joint_names)} joints: {self.joint_names}"
        )

        # Get joint enum class
        self.joint_enum = self._get_joint_enum_class()

    def _get_joint_enum_class(self):
        """Get the appropriate joint enum class for the dataset."""
        if self.dataset_name == "humaneva":
            return GTJointsHumanEVa
        elif self.dataset_name == "movi":
            return GTJointsMoVi
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

    def _get_default_joint_names(self) -> List[str]:
        """Get default joint names for lower body joints (matching config format)."""
        # Default lower body joints matching MoVi config format
        return [
            "LEFT_HIP",
            "RIGHT_HIP",
            "LEFT_KNEE",
            "RIGHT_KNEE",
            "LEFT_ANKLE",
            "RIGHT_ANKLE",
        ]

    def analyze(self, per_frame_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze joint brightness per video across all joints.

        Args:
            per_frame_data: DataFrame with per-frame PCK scores and metadata

        Returns:
            Dict containing analysis results for each video with all joints
        """
        print("Starting per-video joint brightness analysis...")
        print(f"Analyzing {len(self.joint_names)} joints across videos")

        # Group data by video
        video_results = self._analyze_by_video(per_frame_data)

        print(
            f" Per-video joint brightness analysis completed for {len(video_results)} videos"
        )
        return video_results

    def _analyze_by_video(self, per_frame_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze brightness for all joints in each video."""

        # Get grouping columns from config
        grouping_cols = self.config.get_grouping_columns()

        if not grouping_cols:
            print("L No video grouping columns found in configuration")
            return {}

        # Verify that the required columns exist in the data
        missing_cols = [
            col for col in grouping_cols if col not in per_frame_data.columns
        ]
        if missing_cols:
            print(f"L Missing grouping columns in data: {missing_cols}")
            available_cols = [
                col for col in grouping_cols if col in per_frame_data.columns
            ]
            if available_cols:
                print(f"   Using available columns: {available_cols}")
                grouping_cols = available_cols
            else:
                print("   No valid grouping columns found")
                return {}

        print(f"Grouping analysis by: {grouping_cols}")

        # Group by all available grouping columns
        if len(grouping_cols) == 1:
            video_groups = per_frame_data.groupby(grouping_cols[0])
        else:
            video_groups = per_frame_data.groupby(grouping_cols)

        video_results = {}

        for group_key, video_data in video_groups:
            # Create proper video name using config format
            video_name = self.config.create_video_name(
                group_key, grouping_cols)
            print(f"\n--- Processing Video: {video_name} ---")
            # Determine sync offset
            sync_offset = 0
            if hasattr(self.config, "sync_data") and self.config.sync_data:
                try:
                    subject_key = (
                        str(video_data[grouping_cols[0]].iloc[0])
                        if grouping_cols and grouping_cols[0] in video_data.columns
                        else None
                    )
                    action_key = (
                        video_data[grouping_cols[1]].iloc[0]
                        if grouping_cols
                        and len(grouping_cols) > 1
                        and grouping_cols[1] in video_data.columns
                        else None
                    )
                    if action_key and isinstance(action_key, str):
                        action_key = action_key.replace("_", " ").title()
                    camera_index = None
                    if len(grouping_cols) > 2 and grouping_cols[2] in video_data.columns:
                        camera_id = video_data[grouping_cols[2]].iloc[0]
                        try:
                            camera_index = int(camera_id) - 1
                        except Exception:
                            camera_index = 0
                    if subject_key and action_key and camera_index is not None:
                        sync_offset = self.config.sync_data.data["data"][subject_key][
                            action_key
                        ][camera_index]
                    if sync_offset < 0:
                        sync_offset = 0
                    print(f"   Using sync offset: {sync_offset} frames")
                except Exception as e:
                    print(f"   Could not get sync offset: {e}")
                    sync_offset = 0
            # Generate visualization if enabled
            if self.visualize_joints:
                print(
                    f"   Generating joint visualization for {video_name} at sync offset {sync_offset}")
                self.visualize_first_frame_with_joint_circles(
                    video_name, video_data, group_key, grouping_cols,
                    self.visualization_output_dir, sync_offset
                )

            # Analyze all joints for this video
            video_result = self._analyze_video_joints(
                video_name, video_data, group_key, grouping_cols
            )

            if video_result:
                video_results[str(video_name)] = video_result
                print(f"Completed analysis for {video_name}")
            else:
                print(f"Failed analysis for {video_name}")

        return video_results

    def _analyze_video_joints(
        self,
        video_name: str,
        video_data: pd.DataFrame,
        group_key=None,
        grouping_cols: List[str] = None,
    ) -> Dict[str, Any]:
        """Analyze brightness for all joints in a single video, using sync data if available."""

        # Determine sync offset
        sync_offset = 0
        if hasattr(self.config, "sync_data") and self.config.sync_data:
            try:
                subject_key = (
                    str(video_data[grouping_cols[0]].iloc[0])
                    if grouping_cols and grouping_cols[0] in video_data.columns
                    else None
                )
                action_key = (
                    video_data[grouping_cols[1]].iloc[0]
                    if grouping_cols
                    and len(grouping_cols) > 1
                    and grouping_cols[1] in video_data.columns
                    else None
                )
                if action_key and isinstance(action_key, str):
                    action_key = action_key.replace("_", " ").title()
                camera_index = None
                if len(grouping_cols) > 2 and grouping_cols[2] in video_data.columns:
                    camera_id = video_data[grouping_cols[2]].iloc[0]
                    try:
                        camera_index = int(camera_id) - 1
                    except Exception:
                        camera_index = 0
                if subject_key and action_key and camera_index is not None:
                    sync_offset = self.config.sync_data.data["data"][subject_key][
                        action_key
                    ][camera_index]
                if sync_offset < 0:
                    sync_offset = 0
                print(f"   Using sync offset: {sync_offset} frames")
            except Exception as e:
                print(f"   Could not get sync offset: {e}")
                sync_offset = 0

        # Generate visualization if enabled (moved this earlier)
        if self.visualize_joints:
            print(
                f"   Generating joint visualization for {video_name} at sync offset {sync_offset}")
            self.visualize_first_frame_with_joint_circles(
                video_name, video_data, group_key, grouping_cols,
                self.visualization_output_dir, sync_offset
            )

        # Load ground truth coordinates for this video (without sync offset)
        gt_coordinates = self._load_video_ground_truth(
            video_name, video_data, group_key, grouping_cols
        )
        if not gt_coordinates:
            print(f"   No ground truth coordinates found for {video_name}")
            return {}

        # Determine number of frames in ground truth for this video
        num_gt_frames = (
            min([len(coords) for coords in gt_coordinates.values()])
            if gt_coordinates
            else 0
        )

        # Load video file and extract brightness with sync offset
        video_path = self._find_video_path(
            video_name, video_data, group_key, grouping_cols
        )
        if not video_path or not os.path.exists(video_path):
            print(f"   Video file not found for {video_name}")
            return {}

        # Extract brightness for all joints (with sync offset applied during extraction)
        brightness_data = self._extract_video_brightness(
            video_path, gt_coordinates, sync_offset, num_gt_frames
        )
        if not brightness_data:
            print(f"   No brightness data extracted for {video_name}")
            return {}

        # Calculate average brightness for each joint
        avg_brightness = {}
        for joint_name, brightness_values in brightness_data.items():
            valid_values = [v for v in brightness_values if not pd.isna(v)]
            if valid_values:
                avg_brightness[joint_name] = float(np.mean(valid_values))
            else:
                avg_brightness[joint_name] = 0.0

        # Get PCK scores (already averaged per video, so just take the first row)
        pck_scores = {}
        jointwise_pck_columns = [
            col for col in video_data.columns if "jointwise_pck" in col.lower()
        ]

        for pck_column in jointwise_pck_columns:
            joint_name, threshold = self._parse_pck_column_name(pck_column)
            if joint_name in self.joint_names and not video_data[pck_column].empty:
                pck_scores[pck_column] = float(video_data[pck_column].iloc[0])

        # Prepare results
        video_analysis = {
            "video_name": str(video_name),
            "total_frames": len(video_data),
            "synced_frames": num_gt_frames,
            "sync_offset": sync_offset,
            "joints_analyzed": list(brightness_data.keys()),
            "avg_brightness": avg_brightness,
            "pck_scores": pck_scores,
            "brightness_summary": self._get_brightness_summary(brightness_data)
        }

        print(
            f"    Calculated average brightness for {len(avg_brightness)} joints")
        return video_analysis

    def _load_video_ground_truth(
        self,
        video_name: str,
        video_data: pd.DataFrame,
        group_key=None,
        grouping_cols: List[str] = None,
    ) -> Dict[str, np.ndarray]:
        """Load ground truth joint coordinates for a specific video."""

        try:
            # Get ground truth directory from config
            gt_directory = getattr(self.config, "ground_truth_file", None)
            if not gt_directory or not os.path.exists(gt_directory):
                print(f"   Ground truth directory not found: {gt_directory}")
                return {}

            # Find ground truth file for this video
            if self.dataset_name == "movi":
                gt_file = os.path.join(
                    gt_directory, str(group_key), "joints2d_projected.csv"
                )
                if not os.path.exists(gt_file):
                    gt_file = os.path.join(gt_directory, f"{video_name}.csv")

                if not os.path.exists(gt_file):
                    print(f"   Ground truth file not found: {gt_file}")
                    return {}

                print(f"   Loading GT from: {os.path.basename(gt_file)}")

                # Load coordinates based on dataset format
                coordinates = {}
                for joint_name in self.joint_names:
                    joint_coords = self._extract_joint_coordinates(
                        gt_file, joint_name, None, None
                    )
                    if joint_coords is not None:
                        coordinates[joint_name] = joint_coords

            else:  # humaneva
                gt_file = gt_directory

                if not os.path.exists(gt_file):
                    print(f"   Ground truth file not found: {gt_file}")
                    return {}

                print(f"   Loading GT from: {os.path.basename(gt_file)}")

                # For HumanEva, we need to filter the combined GT file by group_key
                coordinates = {}
                for joint_name in self.joint_names:
                    joint_coords = self._extract_joint_coordinates(
                        gt_file, joint_name, group_key, grouping_cols
                    )
                    if joint_coords is not None:
                        coordinates[joint_name] = joint_coords

            print(f"   Loaded coordinates for {len(coordinates)} joints")
            return coordinates

        except Exception as e:
            print(f"   Error loading ground truth for {video_name}: {e}")
            return {}

    def _extract_joint_coordinates(
        self,
        gt_file: str,
        joint_name: str,
        group_key=None,
        grouping_cols: List[str] = None,
    ) -> Optional[np.ndarray]:
        """Extract coordinates for a specific joint from ground truth file."""

        try:
            # Get joint number from enum
            joint_number = self.joint_enum[joint_name].value

            if self.dataset_name == "movi":
                # MoVi format: CSV without headers, reshape to (frames, joints, 2)
                df = pd.read_csv(gt_file, header=None, skiprows=1)
                num_joints = df.shape[1] // 2
                gt_keypoints_np = df.values.reshape((-1, num_joints, 2))

                if joint_number >= num_joints:
                    print(
                        f"   Joint index {joint_number} out of range (max: {num_joints - 1})"
                    )
                    return None

                # Get x and y coordinates for all frames for this joint
                # Shape: (frames, 2)
                joint_coords = gt_keypoints_np[:, joint_number, :]
                return joint_coords

            else:  # humaneva
                # HumanEva format: traditional CSV with column names
                gt_data = pd.read_csv(gt_file)

                # Filter data by group_key if provided
                if group_key is not None and grouping_cols is not None:
                    filter_conditions = []
                    for col, key in zip(grouping_cols, group_key):
                        if col in gt_data.columns:
                            filter_conditions.append(gt_data[col] == key)

                    if filter_conditions:
                        combined_condition = filter_conditions[0]
                        for condition in filter_conditions[1:]:
                            combined_condition &= condition
                        gt_data = gt_data[combined_condition]

                        if gt_data.empty:
                            print(
                                f"   No ground truth data found for {group_key}")
                            return None

                x_col = f"x{joint_number}"
                y_col = f"y{joint_number}"

                if x_col not in gt_data.columns or y_col not in gt_data.columns:
                    print(
                        f"   Joint coordinates not found for joint {joint_number} ({joint_name})"
                    )
                    return None

                # Extract coordinates as numpy array
                x_coords = gt_data[x_col].values
                y_coords = gt_data[y_col].values
                joint_coords = np.column_stack([x_coords, y_coords])
                return joint_coords

        except Exception as e:
            print(f"   Error extracting coordinates for {joint_name}: {e}")
            return None

    def _find_video_path(
        self,
        video_name: str,
        video_data: pd.DataFrame,
        group_key=None,
        grouping_cols: List[str] = None,
    ) -> Optional[str]:
        """Find the path to the video file."""

        try:
            # Get video directory from config
            video_directory = getattr(self.config, "video_directory", None)
            if not video_directory or not os.path.exists(video_directory):
                print(f"   Video directory not found: {video_directory}")
                return None

            # Try different video extensions
            video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".wmv"]

            # For HumanEva, try to construct the original video filename from group_key
            if (
                self.dataset_name == "humaneva"
                and group_key is not None
                and grouping_cols is not None
            ):
                # Extract components from group_key
                if len(grouping_cols) >= 3 and len(group_key) >= 3:
                    subject = str(group_key[0])  # e.g., "S1"
                    action = str(group_key[1])  # e.g., "Walking"
                    camera = str(group_key[2])  # e.g., "C1"

                    # Try different possible filename formats for HumanEva
                    possible_names = [
                        f"{subject}_{action}_{camera}",
                        # The format we use for display
                        f"{subject}_{action}_({camera})",
                        f"{subject}_{action.replace(' ', '_')}_{camera}",
                        f"{subject}_{action.lower()}_{camera}",
                        f"{subject}_{action.lower().replace(' ', '_')}_{camera}",
                    ]

                    for base_name in possible_names:
                        for ext in video_extensions:
                            video_path = os.path.join(
                                video_directory, f"{base_name}{ext}"
                            )
                            if os.path.exists(video_path):
                                return video_path

                # Also try the display name format
                for ext in video_extensions:
                    video_path = os.path.join(
                        video_directory, f"{video_name}{ext}")
                    if os.path.exists(video_path):
                        return video_path

            # For MoVi, try the original format
            elif self.dataset_name == "movi":
                for ext in video_extensions:
                    video_path = os.path.join(
                        video_directory, f"{video_name}_walking_cropped{ext}"
                    )
                    if os.path.exists(video_path):
                        return video_path

            # General fallback: try the video_name directly
            for ext in video_extensions:
                video_path = os.path.join(
                    video_directory, f"{video_name}{ext}")
                if os.path.exists(video_path):
                    return video_path

            # If not found, try recursive search
            for root, dirs, files in os.walk(video_directory):
                for file in files:
                    name, ext = os.path.splitext(file)
                    if ext.lower() in video_extensions:
                        # Check if the filename contains our video components
                        if (
                            self.dataset_name == "humaneva"
                            and group_key is not None
                            and len(group_key) >= 2
                        ):
                            subject = str(group_key[0])
                            action = str(group_key[1])
                            if (
                                subject.lower() in name.lower()
                                and action.lower().replace(" ", "_") in name.lower()
                            ):
                                return os.path.join(root, file)
                        elif str(video_name) in name:
                            return os.path.join(root, file)

            return None

        except Exception as e:
            print(f"   Error finding video path for {video_name}: {e}")
            return None

    def _extract_video_brightness(
        self, video_path: str, gt_coordinates: Dict[str, np.ndarray], sync_offset: int, num_frames: int
    ) -> Dict[str, List[float]]:
        """Extract brightness values for all joints from video with sync offset."""

        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(
                    f"   Failed to open video: {os.path.basename(video_path)}")
                return {}

            total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(
                f"   Processing {num_frames} frames (sync offset: {sync_offset}) for {len(gt_coordinates)} joints"
            )

            # Initialize brightness storage
            brightness_data = {joint: [] for joint in gt_coordinates.keys()}

            # Skip to sync offset
            if sync_offset > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, sync_offset)

            frame_idx = 0
            while frame_idx < num_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                # Convert to LAB color space for perceptually accurate brightness
                lab_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                brightness_frame = lab_frame[:, :, 0]  # L channel (lightness)
                frame_height, frame_width = brightness_frame.shape

                # Extract brightness for each joint in this frame
                for joint_name, joint_coords in gt_coordinates.items():
                    if frame_idx < len(joint_coords):
                        x, y = joint_coords[frame_idx]

                        # Ensure coordinates are within bounds
                        x = int(np.clip(x, 0, frame_width - 1))
                        y = int(np.clip(y, 0, frame_height - 1))

                        # Extract brightness in region around joint
                        brightness = self._sample_brightness_at_point(
                            brightness_frame, x, y, self.sampling_radius
                        )
                        brightness_data[joint_name].append(float(brightness))
                    else:
                        # No ground truth for this frame
                        brightness_data[joint_name].append(np.nan)

                frame_idx += 1

                # Progress indicator
                if frame_idx % 100 == 0:
                    print(f"   Processed {frame_idx}/{num_frames} frames")

            cap.release()
            print(
                f"    Extracted brightness for {len(brightness_data)} joints")
            return brightness_data

        except Exception as e:
            print(f"   Error extracting brightness from video: {e}")
            return {}

    def _sample_brightness_at_point(
        self, brightness_frame: np.ndarray, x: int, y: int, radius: int
    ) -> float:
        """Sample brightness at a point with given radius."""

        frame_height, frame_width = brightness_frame.shape

        # Define sampling region
        x_min = max(0, x - radius)
        x_max = min(frame_width, x + radius + 1)
        y_min = max(0, y - radius)
        y_max = min(frame_height, y + radius + 1)

        # Extract region and calculate mean brightness
        region = brightness_frame[y_min:y_max, x_min:x_max]

        if region.size > 0:
            return np.mean(region)
        else:
            return brightness_frame[y, x]

    def _get_brightness_summary(
        self, brightness_data: Dict[str, List[float]]
    ) -> Dict[str, Any]:
        """Get summary statistics for brightness data across all joints."""

        summary = {}

        for joint_name, brightness_values in brightness_data.items():
            # Filter out NaN values
            valid_values = [v for v in brightness_values if not pd.isna(v)]

            if valid_values:
                summary[joint_name] = {
                    "mean": float(np.mean(valid_values)),
                    "std": float(np.std(valid_values)),
                    "min": float(np.min(valid_values)),
                    "max": float(np.max(valid_values)),
                    "valid_frames": len(valid_values),
                    "total_frames": len(brightness_values),
                }
            else:
                summary[joint_name] = {
                    "mean": 0.0,
                    "std": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                    "valid_frames": 0,
                    "total_frames": len(brightness_values),
                }

        return summary

    def _parse_pck_column_name(self, column_name: str) -> Tuple[str, str]:
        """Parse joint name and threshold from PCK column name."""

        # Example: "left_ankle_jointwise_pck_0.05" -> ("left_ankle", "0.05")
        parts = column_name.split("_")

        # Find 'jointwise' index
        if "jointwise" in parts:
            jointwise_idx = parts.index("jointwise")
            joint_parts = parts[:jointwise_idx]
            joint_name = "_".join(joint_parts)

            # Extract threshold (should be the last part)
            threshold = parts[-1]

            return joint_name, threshold
        else:
            # Fallback parsing
            if "pck" in parts:
                pck_idx = parts.index("pck")
                joint_parts = (
                    parts[: pck_idx - 1]
                    if "jointwise" in parts[pck_idx - 1: pck_idx]
                    else parts[:pck_idx]
                )
                joint_name = "_".join(joint_parts)
                threshold = parts[-1] if len(parts) > pck_idx + \
                    1 else "unknown"
                return joint_name, threshold

        return "unknown_joint", "unknown_threshold"

    def _extract_joint_coordinates(
        self,
        gt_file: str,
        joint_name: str,
        group_key=None,
        grouping_cols: List[str] = None,
    ) -> Optional[np.ndarray]:
        """Extract coordinates for a specific joint from ground truth file.

        Handles both single joint indices and tuple joint indices (for averaging).
        """

        try:
            # Get joint number(s) from enum
            joint_enum_value = self.joint_enum[joint_name].value

            if self.dataset_name == "movi":
                # MoVi format: CSV without headers, reshape to (frames, joints, 2)
                df = pd.read_csv(gt_file, header=None, skiprows=1)
                num_joints = df.shape[1] // 2
                gt_keypoints_np = df.values.reshape((-1, num_joints, 2))

                if isinstance(joint_enum_value, tuple):
                    # Handle tuple joints (e.g., LEFT_KNEE = (12, 13))
                    joint_coords_list = []
                    for joint_idx in joint_enum_value:
                        if joint_idx >= num_joints:
                            print(
                                f"   Joint index {joint_idx} out of range (max: {num_joints - 1})"
                            )
                            continue
                        joint_coords_list.append(
                            gt_keypoints_np[:, joint_idx, :])

                    if not joint_coords_list:
                        return None

                    # Average the coordinates from multiple joints
                    joint_coords = np.mean(joint_coords_list, axis=0)
                    return joint_coords
                else:
                    # Single joint index
                    joint_number = joint_enum_value
                    if joint_number >= num_joints:
                        print(
                            f"   Joint index {joint_number} out of range (max: {num_joints - 1})"
                        )
                        return None

                    # Get x and y coordinates for all frames for this joint
                    # Shape: (frames, 2)
                    joint_coords = gt_keypoints_np[:, joint_number, :]
                    return joint_coords

            else:  # humaneva
                # HumanEva format: traditional CSV with column names
                gt_data = pd.read_csv(gt_file)

                # Filter data by group_key if provided
                if group_key is not None and grouping_cols is not None:
                    subject, action, camera = group_key

                    # Normalize column names (case-insensitive mapping)
                    columns_lower = {c.lower(): c for c in gt_data.columns}

                    filter_conditions = []
                    if "subject" in columns_lower:
                        filter_conditions.append(
                            gt_data[columns_lower["subject"]] == subject)
                    if "action_group" in columns_lower:  # use action_group instead of Action
                        filter_conditions.append(
                            gt_data[columns_lower["action_group"]] == action.replace("_", " "))
                    if "camera" in columns_lower:
                        filter_conditions.append(
                            gt_data[columns_lower["camera"]] == camera)

                    if filter_conditions:
                        combined_condition = filter_conditions[0]
                        for cond in filter_conditions[1:]:
                            combined_condition &= cond

                        gt_data = gt_data[combined_condition]

                        if gt_data.empty:
                            print(
                                f"   No ground truth data found for {group_key}")
                            return None

                if isinstance(joint_enum_value, tuple):
                    # Handle tuple joints (e.g., LEFT_KNEE = (12, 13))
                    joint_coords_list = []
                    for joint_idx in joint_enum_value:
                        x_col = f"x{joint_idx}"
                        y_col = f"y{joint_idx}"

                        if x_col not in gt_data.columns or y_col not in gt_data.columns:
                            print(
                                f"   Joint coordinates not found for joint {joint_idx} in tuple {joint_name}"
                            )
                            continue

                        # Extract coordinates as numpy array
                        x_coords = gt_data[x_col].values
                        y_coords = gt_data[y_col].values
                        joint_coords = np.column_stack([x_coords, y_coords])
                        joint_coords_list.append(joint_coords)

                    if not joint_coords_list:
                        return None

                    # Average the coordinates from multiple joints
                    averaged_coords = np.mean(joint_coords_list, axis=0)
                    return averaged_coords
                else:
                    # Single joint index
                    joint_number = joint_enum_value
                    x_col = f"x{joint_number}"
                    y_col = f"y{joint_number}"

                    if x_col not in gt_data.columns or y_col not in gt_data.columns:
                        print(
                            f"   Joint coordinates not found for joint {joint_number} ({joint_name})"
                        )
                        return None

                    # Extract coordinates as numpy array
                    x_coords = gt_data[x_col].values
                    y_coords = gt_data[y_col].values
                    joint_coords = np.column_stack([x_coords, y_coords])
                    return joint_coords

        except Exception as e:
            print(f"   Error extracting coordinates for {joint_name}: {e}")
            return None

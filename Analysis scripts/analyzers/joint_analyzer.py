"""
Joint Analysis Analyzer

Core analysis logic for joint-wise PCK analysis.
"""

from processors import VideoPathResolver
from utils.config_extractor import extract_analysis_paths
import pandas as pd
import numpy as np
import cv2
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))


class JointAnalyzer:
    """Handles core analysis logic for joint analysis."""

    def __init__(
        self,
        joints_to_analyze: List[str],
        pck_thresholds: List[float],
        joint_enum_class,
        dataset_name: str = None,
    ):
        """Initialize the analyzer.

        Args:
            joints_to_analyze: List of joint names to analyze
            pck_thresholds: List of PCK thresholds to analyze
            joint_enum_class: Enum class containing joint mappings (e.g., GTJointsHumanEVa, GTJointsMoVi)
            dataset_name: Name of the dataset (for path extraction)
        """
        self.joints_to_analyze = joints_to_analyze
        self.pck_thresholds = pck_thresholds
        self.joint_enum_class = joint_enum_class
        self.dataset_name = dataset_name

        # Extract dataset paths and initialize path resolver if available
        self.video_directory = None
        self.ground_truth_directory = None
        self.path_resolver = None

        if dataset_name:
            try:
                paths = extract_analysis_paths(dataset_name)
                self.video_directory = paths.get("video_directory")
                self.ground_truth_directory = paths.get("ground_truth_file")

                # Initialize VideoPathResolver with config-like object
                if paths:
                    from types import SimpleNamespace

                    config = SimpleNamespace()
                    config.name = dataset_name
                    config.video_directory = self.video_directory
                    # Add required columns for different datasets
                    if dataset_name.lower() == "humaneva":
                        config.subject_column = "Subject"
                        config.action_column = "Action"
                        config.camera_column = "Camera"
                    elif dataset_name.lower() == "movi":
                        config.subject_column = "Subject"

                    self.path_resolver = VideoPathResolver(config)
                print(f"Video directory: {self.video_directory}")
                print(f"Ground truth directory: {self.ground_truth_directory}")
            except Exception as e:
                print(f"WARNING: Could not extract paths from config: {e}")

    def _get_joint_number(self, joint_name: str) -> int:
        """Get joint number from enum for the given joint name.

        Args:
            joint_name: Name of the joint (e.g., "LEFT_HIP", "RIGHT_KNEE")

        Returns:
            int: Joint number/index from the enum

        Raises:
            ValueError: If joint name not found in enum
        """
        try:
            joint_enum_value = getattr(self.joint_enum_class, joint_name)

            # Handle tuples (multiple joint indices) by taking the first one
            if isinstance(joint_enum_value.value, tuple):
                return joint_enum_value.value[0]
            else:
                return joint_enum_value.value
        except AttributeError:
            raise ValueError(
                f"Joint '{joint_name}' not found in {self.joint_enum_class.__name__}"
            )

    def extract_brightness_from_video(
        self, pck_data: pd.DataFrame, joint_name: str, sampling_radius: int = 3
    ) -> Optional[np.ndarray]:
        """Extract real brightness values from video using ground truth coordinates.

        Args:
            pck_data: PCK data DataFrame containing video file information
            joint_name: Name of the joint to extract brightness for
            sampling_radius: Radius around joint coordinate for brightness sampling

        Returns:
            np.ndarray: Brightness values or None if extraction fails
        """
        if not self.video_directory or not self.ground_truth_directory:
            print(
                f"WARNING: Video or ground truth directory not available for {joint_name}"
            )
            return None

        brightness_values = []

        # Get joint number from enum
        try:
            joint_number = self._get_joint_number(joint_name)
        except ValueError as e:
            print(f"WARNING: {e}")
            return None

        # Group by video file to process each video
        if "video_file" in pck_data.columns:
            video_groups = pck_data.groupby("video_file")
        elif "subject" in pck_data.columns:
            # Use subject as grouping if video_file not available
            video_groups = pck_data.groupby("subject")
        else:
            print(f"WARNING: No video grouping column found for {joint_name}")
            return None

        for video_name, group_data in video_groups:
            try:
                # Use VideoPathResolver to find video path
                if self.path_resolver:
                    # Create a video row with available data
                    video_row_data = {}
                    if "subject" in group_data.columns:
                        video_row_data["Subject"] = group_data["subject"].iloc[0]
                    if "action" in group_data.columns:
                        video_row_data["Action"] = group_data["action"].iloc[0]
                    if "camera" in group_data.columns:
                        video_row_data["Camera"] = group_data["camera"].iloc[0]

                    # If we have the required data, use path resolver
                    if video_row_data:
                        import pandas as pd

                        video_row = pd.Series(video_row_data)
                        video_path = self.path_resolver.find_video_for_row(
                            video_row)
                    else:
                        # Fallback to simple path construction
                        video_path = os.path.join(
                            self.video_directory, f"{video_name}.mp4"
                        )
                else:
                    # Fallback to simple path construction
                    video_path = os.path.join(
                        self.video_directory, f"{video_name}.mp4")

                if not video_path or not os.path.exists(video_path):
                    # Try alternative extensions for fallback
                    if not self.path_resolver:
                        for ext in [".avi", ".mov", ".mkv"]:
                            alt_path = os.path.join(
                                self.video_directory, f"{video_name}{ext}"
                            )
                            if os.path.exists(alt_path):
                                video_path = alt_path
                                break
                        else:
                            print(
                                f"WARNING: Video file not found for {video_name}")
                            continue
                    else:
                        print(
                            f"WARNING: Video file not found for {video_name}")
                        continue

                # Load ground truth coordinates
                if self.dataset_name.lower() == "movi":
                    gt_file = os.path.join(
                        self.ground_truth_directory,
                        f"{video_name}",
                        "joints2d_projected.csv",
                    )
                elif self.dataset_name.lower() == "humaneva":
                    gt_file = self.ground_truth_directory
                if not os.path.exists(gt_file):
                    # Try alternative naming
                    gt_file = os.path.join(
                        self.ground_truth_directory, f"{video_name}.csv"
                    )
                    if not os.path.exists(gt_file):
                        print(
                            f"WARNING: Ground truth file not found for {video_name}")
                        continue

                # Load ground truth data based on dataset format
                if self.dataset_name.lower() == "movi":
                    # MoVi format: CSV without headers, reshape to (frames, joints, 2)
                    df = pd.read_csv(gt_file, header=None, skiprows=1)
                    num_joints = df.shape[1] // 2
                    gt_keypoints_np = df.values.reshape((-1, num_joints, 2))

                    # Extract coordinates for the specific joint
                    if joint_number >= num_joints:
                        print(
                            f"WARNING: Joint index {joint_number} out of range (max: {num_joints - 1}) in {gt_file}"
                        )
                        continue

                    # Get x and y coordinates for all frames for this joint
                    joint_coords = gt_keypoints_np[
                        :, joint_number, :
                    ]  # Shape: (frames, 2)
                    x_coords = joint_coords[:, 0]
                    y_coords = joint_coords[:, 1]

                    # Create a DataFrame-like structure for compatibility with existing code
                    gt_data = pd.DataFrame({"x": x_coords, "y": y_coords})
                    x_col = "x"
                    y_col = "y"

                else:
                    # HumanEva or other formats: traditional CSV with column names
                    gt_data = pd.read_csv(gt_file)
                    x_col = f"x{joint_number}"
                    y_col = f"y{joint_number}"

                    if x_col not in gt_data.columns or y_col not in gt_data.columns:
                        print(
                            f"WARNING: Joint coordinates not found for joint {joint_number} ({joint_name}) in {gt_file}"
                        )
                        continue

                # Extract brightness from video frames
                print(f"   Processing video: {os.path.basename(video_path)}")
                print(f"   Using ground truth: {os.path.basename(gt_file)}")
                print(f"   Joint coordinates: {x_col}, {y_col}")

                video_brightness = self._extract_brightness_from_frames(
                    video_path, gt_data, x_col, y_col, sampling_radius
                )

                if video_brightness is not None:
                    print(
                        f"   Extracted {len(video_brightness)} brightness values")
                    brightness_values.extend(video_brightness)
                else:
                    print("   No brightness values extracted from this video")

            except Exception as e:
                print(f"WARNING: Error processing video {video_name}: {e}")
                continue

        if brightness_values:
            return np.array(brightness_values)
        else:
            print(f"WARNING: No brightness values extracted for {joint_name}")
            return None

    def _extract_brightness_from_frames(
        self,
        video_path: str,
        gt_data: pd.DataFrame,
        x_col: str,
        y_col: str,
        sampling_radius: int,
    ) -> Optional[List[float]]:
        """Extract brightness values from video frames at specified coordinates.

        Args:
            video_path: Path to video file
            gt_data: Ground truth DataFrame with coordinates
            x_col: Column name for x coordinates
            y_col: Column name for y coordinates
            sampling_radius: Radius for brightness sampling

        Returns:
            List of brightness values or None if extraction fails
        """
        try:
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"ERROR: Could not open video {video_path}")
                return None

            print(
                f"      Video opened successfully: {os.path.basename(video_path)}")
            print(f"      Ground truth data shape: {gt_data.shape}")
            print(f"      Looking for columns: {x_col}, {y_col}")

            brightness_values = []
            frame_idx = 0
            valid_frames = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Check if we have ground truth for this frame
                if frame_idx < len(gt_data):
                    x_coord = gt_data.iloc[frame_idx][x_col]
                    y_coord = gt_data.iloc[frame_idx][y_col]

                    # Skip if coordinates are invalid (NaN or negative)
                    if (
                        pd.isna(x_coord)
                        or pd.isna(y_coord)
                        or x_coord < 0
                        or y_coord < 0
                    ):
                        frame_idx += 1
                        continue

                    # Convert to integer coordinates
                    x, y = int(x_coord), int(y_coord)

                    # Extract brightness at joint location
                    brightness = self._get_brightness_at_point(
                        frame, x, y, sampling_radius
                    )
                    if brightness is not None:
                        brightness_values.append(brightness)
                        valid_frames += 1

                frame_idx += 1

            cap.release()
            print(
                f"      Processed {frame_idx} frames, extracted {valid_frames} valid brightness values"
            )
            return brightness_values if brightness_values else None

        except Exception as e:
            print(f"ERROR: Error extracting brightness from {video_path}: {e}")
            return None

    def _get_brightness_at_point(
        self, frame: np.ndarray, x: int, y: int, radius: int
    ) -> Optional[float]:
        """Extract brightness value at a specific point with given radius.

        Args:
            frame: Video frame (BGR format)
            x: X coordinate
            y: Y coordinate
            radius: Sampling radius around the point

        Returns:
            Average brightness value in LAB L-channel or None if extraction fails
        """
        try:
            # Convert BGR to LAB color space
            lab_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

            # Get frame dimensions
            height, width = lab_frame.shape[:2]

            # Define sampling region with bounds checking
            x_min = max(0, x - radius)
            x_max = min(width, x + radius + 1)
            y_min = max(0, y - radius)
            y_max = min(height, y + radius + 1)

            # Extract L-channel (brightness) from the region
            brightness_region = lab_frame[y_min:y_max, x_min:x_max, 0]

            if brightness_region.size > 0:
                return float(np.mean(brightness_region))
            else:
                return None

        except Exception as e:
            print(
                f"WARNING: Error extracting brightness at point ({x}, {y}): {e}")
            return None

    def calculate_statistics(self, values: np.ndarray) -> Dict[str, float]:
        """Calculate statistical measures for a set of values.

        Args:
            values: Array of values to analyze

        Returns:
            dict: Dictionary of statistical measures
        """
        if len(values) == 0:
            return {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "median": 0.0,
                "count": 0,
            }

        q75, q25 = np.percentile(values, [75, 25])

        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "median": float(np.median(values)),
            "iqr": float(q75 - q25),
            "count": len(values),
        }

    def analyze_joint_threshold(
        self,
        pck_data: pd.DataFrame,
        joint_name: str,
        threshold: float,
        joint_index: int,
    ) -> Dict[str, Any]:
        """Analyze a specific joint at a specific threshold.

        Args:
            pck_data: PCK data DataFrame
            joint_name: Name of the joint to analyze
            threshold: PCK threshold to analyze
            joint_index: Index of the joint (for fallback simulation)

        Returns:
            dict: Analysis results for this joint-threshold combination
        """
        metric_name = f"{joint_name}_jointwise_pck_{threshold:g}"

        if metric_name not in pck_data.columns:
            return None

        # Extract PCK scores
        pck_scores = pck_data[metric_name].dropna().values

        if len(pck_scores) == 0:
            return None

        # Try to extract real brightness values from videos
        brightness_values = self.extract_brightness_from_video(
            pck_data, joint_name)

        # If real extraction fails, fall back to simulation with warning
        if brightness_values is None:
            print(
                "No Ground truth data"
            )

        else:
            print(
                f"✓ Extracted {len(brightness_values)} real brightness values for {joint_name}"
            )
            # Ensure brightness and PCK arrays have same length
            min_length = min(len(brightness_values), len(pck_scores))
            brightness_values = brightness_values[:min_length]
            pck_scores = pck_scores[:min_length]

        # Calculate statistics
        pck_stats = self.calculate_statistics(pck_scores)
        brightness_stats = self.calculate_statistics(brightness_values)

        # Calculate correlation
        correlation = (
            np.corrcoef(brightness_values, pck_scores)[0, 1]
            if len(pck_scores) > 1
            else 0.0
        )

        return {
            "joint": joint_name,
            "threshold": threshold,
            "pck_scores": pck_scores,
            "brightness_values": brightness_values,
            "pck_stats": pck_stats,
            "brightness_stats": brightness_stats,
            "correlation": correlation,
            "avg_pck": pck_stats["mean"],
            "avg_brightness": brightness_stats["mean"],
        }

    def _generate_simulated_brightness(
        self, pck_scores: np.ndarray, joint_index: int = 0
    ) -> np.ndarray:
        """Generate simulated brightness values as fallback when real extraction fails.

        Args:
            pck_scores: Array of PCK scores
            joint_index: Index of the joint for variation

        Returns:
            np.ndarray: Simulated brightness values
        """
        # Use joint-specific seed for variation
        np.random.seed(42 + joint_index)
        brightness_values = np.random.normal(100, 30, len(pck_scores))
        brightness_values = np.clip(brightness_values, 0, 255)

        # Add correlation between brightness and PCK for demonstration
        correlation_noise = np.random.normal(0, 0.1, len(pck_scores))
        brightness_values += (pck_scores * 50) + correlation_noise
        brightness_values = np.clip(brightness_values, 0, 255)

        return brightness_values

    def run_complete_analysis(self, pck_data: pd.DataFrame) -> Dict[str, Any]:
        """Run complete joint analysis for all joints and thresholds.

        Args:
            pck_data: PCK data DataFrame

        Returns:
            dict: Complete analysis results
        """
        print("Running joint analysis...")

        analysis_results = {}

        for j, joint_name in enumerate(self.joints_to_analyze):
            for threshold in self.pck_thresholds:
                print(f"  Analyzing {joint_name} at threshold {threshold}")

                result = self.analyze_joint_threshold(
                    pck_data, joint_name, threshold, j
                )

                if result is not None:
                    metric_key = f"{joint_name}_jointwise_pck_{threshold:g}"
                    analysis_results[metric_key] = result
                else:
                    print(
                        f"    WARNING: No data for {joint_name} at threshold {threshold}"
                    )

        print(
            f"Analysis completed. Generated {len(analysis_results)} results.")
        return analysis_results

    def get_average_data_for_plotting(
        self, analysis_results: Dict[str, Any]
    ) -> Dict[str, Dict]:
        """Extract average data organized by threshold for plotting.

        Args:
            analysis_results: Complete analysis results

        Returns:
            dict: Data organized by threshold for plotting
        """
        threshold_data = {}

        # Group by threshold
        for threshold in self.pck_thresholds:
            joint_data = {
                "joint_names": [],
                "avg_brightness": [],
                "avg_pck": [],
                "colors": ["red", "blue", "green", "orange", "purple", "brown"],
            }

            for j, joint_name in enumerate(self.joints_to_analyze):
                metric_key = f"{joint_name}_jointwise_pck_{threshold:g}"

                if metric_key in analysis_results:
                    result = analysis_results[metric_key]
                    joint_data["joint_names"].append(
                        joint_name.replace("_", " "))
                    joint_data["avg_brightness"].append(
                        result["avg_brightness"])
                    joint_data["avg_pck"].append(result["avg_pck"])

            threshold_data[threshold] = joint_data

        return threshold_data

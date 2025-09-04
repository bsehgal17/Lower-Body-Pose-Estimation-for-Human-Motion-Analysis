"""
Joint Brightness Analyzer.

Analyzes brightness values at joint coordinates per frame from video data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
from core.base_classes import BaseAnalyzer
import os
import cv2


class JointBrightnessAnalyzer(BaseAnalyzer):
    """Analyzer for joint brightness per frame analysis."""

    def __init__(self, config, joint_names: List[str] = None, sampling_radius: int = 3):
        """
        Initialize joint brightness analyzer.

        Args:
            config: Analysis configuration
            joint_names: List of joint names to analyze (from enum)
            sampling_radius: Radius around joint for brightness sampling
        """
        super().__init__(config)
        self.joint_names = joint_names or self._get_default_joint_names()
        self.sampling_radius = sampling_radius

    def _get_default_joint_names(self) -> List[str]:
        """Get default joint names based on the PCK columns in data."""
        # Extract joint names from PCK column patterns
        # LEFT_HIP_jointwise_pck_0.01 -> LEFT_HIP
        default_joints = [
            "LEFT_HIP",
            "RIGHT_HIP",
            "LEFT_KNEE",
            "RIGHT_KNEE",
            "LEFT_ANKLE",
            "RIGHT_ANKLE",
        ]
        return default_joints

    def analyze(self, per_frame_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze joint brightness from per-frame data.

        Args:
            per_frame_data: DataFrame with per-frame PCK scores and metadata

        Returns:
            Dict containing analysis results for each joint and PCK threshold
        """
        print(
            f"Starting joint brightness analysis for {len(self.joint_names)} joints..."
        )

        # Load ground truth coordinates
        gt_coordinates = self._load_ground_truth_coordinates(per_frame_data)
        if not gt_coordinates:
            print("❌ No ground truth coordinates found")
            return {}

        # Load video data and extract brightness
        brightness_data = self._extract_brightness_from_video(
            per_frame_data, gt_coordinates
        )
        if not brightness_data:
            print("❌ No brightness data extracted")
            return {}

        # Analyze jointwise PCK scores with brightness
        analysis_results = self._analyze_jointwise_pck_brightness(
            per_frame_data, brightness_data
        )

        print(
            f"✅ Joint brightness analysis completed for {len(analysis_results)} metrics"
        )
        return analysis_results

    def _load_ground_truth_coordinates(
        self, per_frame_data: pd.DataFrame
    ) -> Dict[str, np.ndarray]:
        """Load ground truth joint coordinates from dataset."""
        print("Loading ground truth coordinates...")

        try:
            # Import ground truth loader
            import sys

            sys.path.append(
                os.path.join(os.path.dirname(__file__), "..", "ground_truth_analysis")
            )
            from gt_data_loader import GroundTruthDataLoader

            # Get dataset name from config
            dataset_name = getattr(self.config, "name", "humaneva")
            gt_loader = GroundTruthDataLoader(dataset_name)

            # Extract coordinates for our joints
            coordinates = gt_loader.extract_joint_coordinates(self.joint_names)

            if coordinates:
                print(f"✅ Loaded coordinates for {len(coordinates)} joints")
                return coordinates
            else:
                print("❌ No coordinates loaded")
                return {}

        except Exception as e:
            print(f"❌ Error loading ground truth coordinates: {e}")
            return {}

    def _extract_brightness_from_video(
        self, per_frame_data: pd.DataFrame, gt_coordinates: Dict[str, np.ndarray]
    ) -> Dict[str, List[float]]:
        """Extract brightness values from video at joint coordinates."""
        print("Extracting brightness from video...")

        try:
            # Get video path from config
            video_folder = getattr(self.config, "video_directory", None)
            if not video_folder or not os.path.exists(video_folder):
                print(f"❌ Video directory not found: {video_folder}")
                return {}

            # Find video files
            video_files = self._find_video_files(video_folder)
            if not video_files:
                print("❌ No video files found")
                return {}

            # Use first video file for now
            video_path = video_files[0]
            print(f"Using video: {os.path.basename(video_path)}")

            # Extract brightness at coordinates
            brightness_data = self._extract_brightness_at_coordinates(
                video_path, gt_coordinates
            )

            return brightness_data

        except Exception as e:
            print(f"❌ Error extracting brightness: {e}")
            return {}

    def _find_video_files(self, video_folder: str) -> List[str]:
        """Find video files in the directory."""
        video_extensions = (".mp4", ".avi", ".mov", ".mkv", ".wmv")
        video_files = []

        for root, dirs, files in os.walk(video_folder):
            for file in files:
                if file.lower().endswith(video_extensions):
                    video_files.append(os.path.join(root, file))

        return video_files

    def _extract_brightness_at_coordinates(
        self, video_path: str, coordinates: Dict[str, np.ndarray]
    ) -> Dict[str, List[float]]:
        """Extract brightness values at joint coordinates from video."""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"❌ Failed to open video: {video_path}")
                return {}

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"   Processing {total_frames} frames...")

            # Initialize brightness storage
            brightness_data = {joint: [] for joint in coordinates.keys()}

            frame_idx = 0
            while frame_idx < total_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                # Convert to grayscale
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame_height, frame_width = gray_frame.shape

                # Extract brightness for each joint
                for joint_name, joint_coords in coordinates.items():
                    if frame_idx < len(joint_coords):
                        x, y = joint_coords[frame_idx]

                        # Ensure coordinates are within bounds
                        x = int(np.clip(x, 0, frame_width - 1))
                        y = int(np.clip(y, 0, frame_height - 1))

                        # Extract brightness in region around joint
                        x_min = max(0, x - self.sampling_radius)
                        x_max = min(frame_width, x + self.sampling_radius + 1)
                        y_min = max(0, y - self.sampling_radius)
                        y_max = min(frame_height, y + self.sampling_radius + 1)

                        region = gray_frame[y_min:y_max, x_min:x_max]

                        if region.size > 0:
                            brightness = np.mean(region)
                        else:
                            brightness = gray_frame[y, x]

                        brightness_data[joint_name].append(float(brightness))
                    else:
                        brightness_data[joint_name].append(np.nan)

                frame_idx += 1

                if frame_idx % 100 == 0:
                    print(f"   Processed {frame_idx}/{total_frames} frames")

            cap.release()
            print(f"✅ Extracted brightness for {len(brightness_data)} joints")
            return brightness_data

        except Exception as e:
            print(f"❌ Error extracting brightness at coordinates: {e}")
            return {}

    def _analyze_jointwise_pck_brightness(
        self, per_frame_data: pd.DataFrame, brightness_data: Dict[str, List[float]]
    ) -> Dict[str, Any]:
        """Analyze relationship between jointwise PCK scores and brightness."""
        analysis_results = {}

        # Find jointwise PCK columns
        jointwise_pck_columns = [
            col for col in per_frame_data.columns if "jointwise_pck" in col.lower()
        ]

        if not jointwise_pck_columns:
            print("❌ No jointwise PCK columns found")
            return {}

        print(f"Found {len(jointwise_pck_columns)} jointwise PCK columns")

        for pck_column in jointwise_pck_columns:
            print(f"Analyzing {pck_column}...")

            # Extract joint name and threshold from column name
            joint_name, threshold = self._parse_pck_column_name(pck_column)

            if joint_name not in brightness_data:
                print(f"   Skipping {pck_column} - no brightness data for {joint_name}")
                continue

            # Get PCK scores and brightness values
            pck_scores = per_frame_data[pck_column].dropna()
            joint_brightness = brightness_data[joint_name]

            # Align data lengths
            min_length = min(len(pck_scores), len(joint_brightness))
            if min_length == 0:
                continue

            pck_scores = pck_scores.iloc[:min_length]
            joint_brightness = joint_brightness[:min_length]

            # Remove NaN values
            valid_mask = ~(pd.isna(pck_scores) | pd.isna(joint_brightness))
            pck_scores = pck_scores[valid_mask]
            joint_brightness = np.array(joint_brightness)[valid_mask]

            if len(pck_scores) == 0:
                continue

            # Analyze brightness distribution by PCK score
            result = self._analyze_pck_brightness_distribution(
                pck_scores, joint_brightness
            )
            result["joint_name"] = joint_name
            result["threshold"] = threshold
            result["total_frames"] = len(pck_scores)

            analysis_results[pck_column] = result

        return analysis_results

    def _parse_pck_column_name(self, column_name: str) -> tuple:
        """Parse joint name and threshold from PCK column name."""
        # Format: LEFT_HIP_jointwise_pck_0.01
        parts = column_name.split("_")

        # Find the threshold (last part after splitting by _)
        threshold = parts[-1] if parts else "0.05"

        # Joint name is everything before 'jointwise'
        joint_parts = []
        for part in parts:
            if part.lower() == "jointwise":
                break
            joint_parts.append(part)

        joint_name = "_".join(joint_parts) if joint_parts else "UNKNOWN"

        return joint_name, threshold

    def _analyze_pck_brightness_distribution(
        self, pck_scores: pd.Series, brightness_values: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze brightness distribution for different PCK score ranges."""
        result = {
            "pck_scores": list(pck_scores),
            "brightness_values": list(brightness_values),
            "brightness_stats": {},
            "correlation": {},
            "score_ranges": {},
        }

        # Overall statistics
        result["brightness_stats"]["overall"] = {
            "mean": float(np.mean(brightness_values)),
            "std": float(np.std(brightness_values)),
            "min": float(np.min(brightness_values)),
            "max": float(np.max(brightness_values)),
            "median": float(np.median(brightness_values)),
        }

        # Correlation analysis
        try:
            correlation = np.corrcoef(pck_scores, brightness_values)[0, 1]
            result["correlation"]["pearson"] = (
                float(correlation) if not np.isnan(correlation) else 0.0
            )
        except Exception:
            result["correlation"]["pearson"] = 0.0

        # Score range analysis
        score_ranges = [("low", 0.0, 0.3), ("medium", 0.3, 0.7), ("high", 0.7, 1.0)]

        for range_name, min_score, max_score in score_ranges:
            mask = (pck_scores >= min_score) & (pck_scores < max_score)
            if np.any(mask):
                range_brightness = brightness_values[mask]
                result["score_ranges"][range_name] = {
                    "count": int(np.sum(mask)),
                    "mean_brightness": float(np.mean(range_brightness)),
                    "std_brightness": float(np.std(range_brightness)),
                }

        return result

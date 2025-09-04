"""
Enhanced Ground Truth Data Loader for Joint Analysis

Handles loading and processing ground truth joint coordinates from various dataset formats.
Supports extracting specific joint coordinates for brightness analysis.
Works with Excel evaluation results and original dataset files.
"""

import json
import numpy as np
import pandas as pd
import os
from typing import Dict, List, Optional, Any
import h5py
from pathlib import Path


class GroundTruthDataLoader:
    """Enhanced loader for ground truth joint coordinate data."""

    def __init__(self, dataset_name: str, gt_file_path: str = None):
        """
        Initialize ground truth data loader.

        Args:
            dataset_name: Name of the dataset (movi, humaneva, etc.)
            gt_file_path: Optional custom path to ground truth file
        """
        self.dataset_name = dataset_name.lower()
        self.gt_file_path = gt_file_path
        self.joint_mapping = self._get_joint_mapping()
        self.data = None

    def _get_joint_mapping(self) -> Dict[str, Any]:
        """Get joint name mapping for the dataset."""
        mappings = {
            "movi": {
                "LEFT_HIP": ["left_hip", "L_Hip", "leftHip", 0, "x0", "y0"],
                "RIGHT_HIP": ["right_hip", "R_Hip", "rightHip", 1, "x1", "y1"],
                "LEFT_KNEE": ["left_knee", "L_Knee", "leftKnee", 2, "x2", "y2"],
                "RIGHT_KNEE": ["right_knee", "R_Knee", "rightKnee", 3, "x3", "y3"],
                "LEFT_ANKLE": ["left_ankle", "L_Ankle", "leftAnkle", 4, "x4", "y4"],
                "RIGHT_ANKLE": ["right_ankle", "R_Ankle", "rightAnkle", 5, "x5", "y5"],
                "LEFT_FOOT": ["left_foot", "L_Foot", "leftFoot", 6, "x6", "y6"],
                "RIGHT_FOOT": ["right_foot", "R_Foot", "rightFoot", 7, "x7", "y7"],
            },
            "humaneva": {
                "LEFT_HIP": ["L_Hip", "left_hip", "LeftHip", 0, "x0", "y0"],
                "RIGHT_HIP": ["R_Hip", "right_hip", "RightHip", 1, "x1", "y1"],
                "LEFT_KNEE": ["L_Knee", "left_knee", "LeftKnee", 2, "x2", "y2"],
                "RIGHT_KNEE": ["R_Knee", "right_knee", "RightKnee", 3, "x3", "y3"],
                "LEFT_ANKLE": ["L_Ankle", "left_ankle", "LeftAnkle", 4, "x4", "y4"],
                "RIGHT_ANKLE": ["R_Ankle", "right_ankle", "RightAnkle", 5, "x5", "y5"],
                "LEFT_FOOT": ["L_Foot", "left_foot", "LeftFoot", 6, "x6", "y6"],
                "RIGHT_FOOT": ["R_Foot", "right_foot", "RightFoot", 7, "x7", "y7"],
            },
        }
        return mappings.get(self.dataset_name, mappings["movi"])

    def load_ground_truth_data(self) -> bool:
        """Load ground truth data from file."""
        if not self.gt_file_path or not os.path.exists(self.gt_file_path):
            print(f"❌ Ground truth file not found: {self.gt_file_path}")
            return False

        try:
            file_ext = Path(self.gt_file_path).suffix.lower()

            if file_ext == ".json":
                self.data = self._load_json_data()
            elif file_ext == ".h5" or file_ext == ".hdf5":
                self.data = self._load_h5_data()
            elif file_ext == ".csv":
                self.data = self._load_csv_data()
            elif file_ext == ".npz":
                self.data = self._load_npz_data()
            elif file_ext in [".xlsx", ".xls"]:
                self.data = self._load_excel_data()
            else:
                print(f"❌ Unsupported file format: {file_ext}")
                return False

            if self.data is not None:
                print(f"✅ Loaded ground truth data from {self.gt_file_path}")
                return True
            else:
                print(f"❌ Failed to load data from {self.gt_file_path}")
                return False

        except Exception as e:
            print(f"❌ Error loading ground truth data: {e}")
            return False

    def _load_json_data(self) -> Optional[Dict]:
        """Load data from JSON file."""
        with open(self.gt_file_path, "r") as f:
            return json.load(f)

    def _load_h5_data(self) -> Optional[Dict]:
        """Load data from HDF5 file."""
        data = {}
        with h5py.File(self.gt_file_path, "r") as f:

            def extract_datasets(name, obj):
                if isinstance(obj, h5py.Dataset):
                    data[name] = obj[:]

            f.visititems(extract_datasets)
        return data if data else None

    def _load_csv_data(self) -> Optional[pd.DataFrame]:
        """Load data from CSV file."""
        return pd.read_csv(self.gt_file_path)

    def _load_npz_data(self) -> Optional[Dict]:
        """Load data from NPZ file."""
        return dict(np.load(self.gt_file_path))

    def _load_excel_data(self) -> Optional[Dict]:
        """Load data from Excel file (evaluation results)."""
        try:
            # Try to read all sheets
            excel_data = pd.read_excel(self.gt_file_path, sheet_name=None)

            # Priority order for sheets containing ground truth data
            sheet_priority = [
                "Per-Frame Scores",
                "Jointwise Metrics",
                "Overall Metrics",
            ]

            for sheet_name in sheet_priority:
                if sheet_name in excel_data:
                    return {"sheet_data": excel_data, "primary_sheet": sheet_name}

            # If no priority sheets found, use the first available sheet
            if excel_data:
                first_sheet = list(excel_data.keys())[0]
                return {"sheet_data": excel_data, "primary_sheet": first_sheet}

            return None

        except Exception as e:
            print(f"❌ Error loading Excel file: {e}")
            return None

    def extract_joint_coordinates(
        self, joint_names: List[str]
    ) -> Dict[str, np.ndarray]:
        """
        Extract coordinates for specified joints.

        Args:
            joint_names: List of joint names to extract

        Returns:
            Dictionary mapping joint names to coordinate arrays [N_frames, 2] (x, y)
        """
        if self.data is None:
            if not self.load_ground_truth_data():
                return {}

        coordinates = {}

        # Check if this is evaluation data (from PCK analysis pipeline)
        if self._is_evaluation_data():
            coordinates = self._extract_from_evaluation_data(joint_names)
        else:
            # Original dataset files
            for joint_name in joint_names:
                joint_coords = self._extract_single_joint_coordinates(joint_name)
                if joint_coords is not None:
                    coordinates[joint_name] = joint_coords
                else:
                    print(f"⚠️  Could not extract coordinates for joint: {joint_name}")

        print(
            f"✅ Extracted coordinates for {len(coordinates)}/{len(joint_names)} joints"
        )
        return coordinates

    def _is_evaluation_data(self) -> bool:
        """Check if the loaded data is from evaluation pipeline (Excel with PCK results)."""
        return (
            isinstance(self.data, dict)
            and "sheet_data" in self.data
            and "primary_sheet" in self.data
        )

    def _extract_from_evaluation_data(
        self, joint_names: List[str]
    ) -> Dict[str, np.ndarray]:
        """Extract ground truth coordinates from evaluation pipeline data."""
        try:
            # This is a placeholder - in the actual implementation, we would need to
            # access the original ground truth data that was used to generate the PCK scores
            print(
                "⚠️  Evaluation data detected, but ground truth coordinates extraction"
            )
            print("   from PCK results is not yet implemented. Please provide original")
            print(
                "   ground truth files (CSV, JSON, HDF5) instead of evaluation results."
            )

            # For now, we'll try to extract what we can from the evaluation data
            # but this won't give us actual coordinates
            sheet_data = self.data["sheet_data"]
            primary_sheet = self.data["primary_sheet"]

            df = sheet_data[primary_sheet]

            # Look for video metadata that might help us locate original GT files
            coordinates = {}
            if "subject" in df.columns and "action" in df.columns:
                # This suggests HumanEva data
                coordinates = self._extract_from_humaneva_evaluation(df, joint_names)
            elif any(col in df.columns for col in ["video_id", "segment", "sequence"]):
                # This suggests MoVi data
                coordinates = self._extract_from_movi_evaluation(df, joint_names)

            return coordinates

        except Exception as e:
            print(f"❌ Error extracting from evaluation data: {e}")
            return {}

    def _extract_from_humaneva_evaluation(
        self, df: pd.DataFrame, joint_names: List[str]
    ) -> Dict[str, np.ndarray]:
        """Extract coordinates from HumanEva evaluation data."""
        print("🔍 Attempting to locate original HumanEva ground truth data...")

        # Try to find the original CSV files based on the evaluation metadata
        coordinates = {}

        # Get unique combinations of subject/action/camera from the evaluation data
        if all(col in df.columns for col in ["subject", "action", "camera"]):
            unique_combinations = df[["subject", "action", "camera"]].drop_duplicates()

            for _, row in unique_combinations.iterrows():
                subject = row["subject"]
                action = row["action"]
                camera = row["camera"]

                # Try to find the original ground truth CSV
                gt_csv_path = self._find_humaneva_gt_csv(subject, action, camera)
                if gt_csv_path and os.path.exists(gt_csv_path):
                    print(f"   Found GT file: {gt_csv_path}")

                    # Load the original ground truth data
                    try:
                        from dataset_files.HumanEva.get_gt_keypoint import (
                            GroundTruthLoader,
                        )

                        gt_loader = GroundTruthLoader(gt_csv_path)
                        gt_keypoints = gt_loader.get_keypoints(subject, action, camera)

                        # Extract coordinates for each joint
                        for joint_name in joint_names:
                            if joint_name in self.joint_mapping:
                                joint_idx = None
                                possible_indices = [
                                    idx
                                    for idx in self.joint_mapping[joint_name]
                                    if isinstance(idx, int)
                                ]
                                if possible_indices:
                                    joint_idx = possible_indices[0]
                                    if joint_idx < gt_keypoints.shape[1]:
                                        if joint_name not in coordinates:
                                            coordinates[joint_name] = gt_keypoints[
                                                :, joint_idx, :
                                            ]
                                        else:
                                            # Concatenate if we have multiple sequences
                                            coordinates[joint_name] = np.vstack(
                                                [
                                                    coordinates[joint_name],
                                                    gt_keypoints[:, joint_idx, :],
                                                ]
                                            )
                    except Exception as e:
                        print(f"   ⚠️  Error loading {gt_csv_path}: {e}")

        return coordinates

    def _extract_from_movi_evaluation(
        self, df: pd.DataFrame, joint_names: List[str]
    ) -> Dict[str, np.ndarray]:
        """Extract coordinates from MoVi evaluation data."""
        print("🔍 Attempting to locate original MoVi ground truth data...")

        # Try to find the original CSV/MAT files based on the evaluation metadata
        coordinates = {}

        # Look for video/sequence identifiers
        video_cols = [
            col
            for col in df.columns
            if "video" in col.lower() or "sequence" in col.lower()
        ]

        if video_cols:
            unique_videos = df[video_cols[0]].drop_duplicates()

            for video_id in unique_videos:
                # Try to find the corresponding ground truth file
                gt_file_path = self._find_movi_gt_file(video_id)
                if gt_file_path and os.path.exists(gt_file_path):
                    print(f"   Found GT file: {gt_file_path}")

                    try:
                        if gt_file_path.endswith(".csv"):
                            # Load CSV format (joints2d_projected.csv)
                            gt_df = pd.read_csv(gt_file_path, header=None, skiprows=1)
                            gt_array = gt_df.values.reshape(-1, len(joint_names), 2)

                            for i, joint_name in enumerate(joint_names):
                                if joint_name not in coordinates:
                                    coordinates[joint_name] = gt_array[:, i, :]
                                else:
                                    coordinates[joint_name] = np.vstack(
                                        [coordinates[joint_name], gt_array[:, i, :]]
                                    )

                        elif gt_file_path.endswith(".mat"):
                            # Load MAT format
                            from dataset_files.MoVi.movi_gt_loader import (
                                MoViGroundTruthLoader,
                            )

                            gt_loader = MoViGroundTruthLoader(gt_file_path)
                            gt_keypoints = gt_loader.get_keypoints()

                            for i, joint_name in enumerate(joint_names):
                                if i < gt_keypoints.shape[1]:
                                    if joint_name not in coordinates:
                                        coordinates[joint_name] = gt_keypoints[:, i, :]
                                    else:
                                        coordinates[joint_name] = np.vstack(
                                            [
                                                coordinates[joint_name],
                                                gt_keypoints[:, i, :],
                                            ]
                                        )
                    except Exception as e:
                        print(f"   ⚠️  Error loading {gt_file_path}: {e}")

        return coordinates

    def _find_humaneva_gt_csv(
        self, subject: str, action: str, camera: int
    ) -> Optional[str]:
        """Find the original HumanEva ground truth CSV file."""
        # Common paths where HumanEva ground truth might be located
        possible_base_paths = [
            os.path.dirname(self.gt_file_path),
            os.path.join(os.path.dirname(self.gt_file_path), "..", "ground_truth"),
            os.path.join(os.path.dirname(self.gt_file_path), "..", "..", "HumanEva"),
        ]

        for base_path in possible_base_paths:
            if not os.path.exists(base_path):
                continue

            # Look for combined CSV or individual files
            combined_csv = os.path.join(base_path, "combined_humaneva_data.csv")
            if os.path.exists(combined_csv):
                return combined_csv

            # Look for individual files
            action_safe = action.replace(" ", "_")
            individual_csv = os.path.join(
                base_path, f"{subject}_{action_safe}_C{camera + 1}.csv"
            )
            if os.path.exists(individual_csv):
                return individual_csv

        return None

    def _find_movi_gt_file(self, video_id: str) -> Optional[str]:
        """Find the original MoVi ground truth file."""
        # Common paths where MoVi ground truth might be located
        possible_base_paths = [
            os.path.dirname(self.gt_file_path),
            os.path.join(os.path.dirname(self.gt_file_path), "..", "ground_truth"),
            os.path.join(os.path.dirname(self.gt_file_path), "..", "..", "MoVi"),
        ]

        for base_path in possible_base_paths:
            if not os.path.exists(base_path):
                continue

            # Look for CSV files
            csv_file = os.path.join(base_path, video_id, "joints2d_projected.csv")
            if os.path.exists(csv_file):
                return csv_file

            # Look for MAT files
            mat_file = os.path.join(base_path, video_id, f"{video_id}.mat")
            if os.path.exists(mat_file):
                return mat_file

        return None

    def _extract_single_joint_coordinates(
        self, joint_name: str
    ) -> Optional[np.ndarray]:
        """Extract coordinates for a single joint from original dataset files."""
        if joint_name not in self.joint_mapping:
            print(f"⚠️  Unknown joint: {joint_name}")
            return None

        possible_names = self.joint_mapping[joint_name]

        # Try different data structures
        coordinates = None

        # Method 1: Direct key lookup in dictionary/JSON
        if isinstance(self.data, dict):
            coordinates = self._find_coordinates_in_dict(possible_names)

        # Method 2: DataFrame with column names
        elif isinstance(self.data, pd.DataFrame):
            coordinates = self._find_coordinates_in_dataframe(possible_names)

        # Method 3: Array-based lookup with indices
        if coordinates is None:
            coordinates = self._find_coordinates_by_index(possible_names)

        # Method 4: Try nested structures
        if coordinates is None:
            coordinates = self._find_coordinates_nested(possible_names)

        return coordinates

    def _find_coordinates_in_dict(self, possible_names: List) -> Optional[np.ndarray]:
        """Find coordinates in dictionary structure."""
        for name in possible_names:
            if isinstance(name, str) and name in self.data:
                coords = self.data[name]
                return self._normalize_coordinates(coords)
        return None

    def _find_coordinates_in_dataframe(
        self, possible_names: List
    ) -> Optional[np.ndarray]:
        """Find coordinates in DataFrame structure."""
        df = self.data

        # Look for x, y columns with various naming conventions
        for name in possible_names:
            if isinstance(name, str):
                # Try different naming patterns
                patterns = [
                    (f"{name}_x", f"{name}_y"),
                    (f"x_{name}", f"y_{name}"),
                    (name + "_x", name + "_y"),
                    (f"x{name}", f"y{name}"),
                ]

                for x_col, y_col in patterns:
                    if x_col in df.columns and y_col in df.columns:
                        x_coords = df[x_col].values
                        y_coords = df[y_col].values
                        return np.column_stack([x_coords, y_coords])

                # Try direct column lookup
                if name in df.columns:
                    coords = df[name].values
                    return self._normalize_coordinates(coords)

        # Try numeric column names (x0, y0, x1, y1, etc.)
        x_cols = [
            name
            for name in possible_names
            if isinstance(name, str) and name.startswith("x")
        ]
        y_cols = [
            name
            for name in possible_names
            if isinstance(name, str) and name.startswith("y")
        ]

        for x_col, y_col in zip(x_cols, y_cols):
            if x_col in df.columns and y_col in df.columns:
                x_coords = df[x_col].values
                y_coords = df[y_col].values
                return np.column_stack([x_coords, y_coords])

        return None

    def _find_coordinates_by_index(self, possible_names: List) -> Optional[np.ndarray]:
        """Find coordinates using numeric indices."""
        # Look for numeric indices in possible names
        indices = [name for name in possible_names if isinstance(name, int)]

        if not indices:
            return None

        # Try to find array-like data
        arrays = self._find_array_data()

        for array_name, array_data in arrays.items():
            if array_data.ndim >= 2:
                for idx in indices:
                    if idx < array_data.shape[-1]:
                        # Extract coordinates for this joint index
                        if array_data.ndim == 3:  # [frames, joints, 2]
                            return array_data[:, idx, :2]
                        elif array_data.ndim == 2:  # [frames, features]
                            # Assume alternating x, y coordinates
                            if idx * 2 + 1 < array_data.shape[1]:
                                x_coords = array_data[:, idx * 2]
                                y_coords = array_data[:, idx * 2 + 1]
                                return np.column_stack([x_coords, y_coords])

        return None

    def _find_coordinates_nested(self, possible_names: List) -> Optional[np.ndarray]:
        """Find coordinates in nested structure."""
        if not isinstance(self.data, dict):
            return None

        # Try common nested structures
        nested_keys = ["joints", "keypoints", "landmarks", "poses", "annotations"]

        for key in nested_keys:
            if key in self.data:
                nested_data = self.data[key]

                # Recursive search in nested data
                if isinstance(nested_data, dict):
                    for name in possible_names:
                        if isinstance(name, str) and name in nested_data:
                            coords = nested_data[name]
                            return self._normalize_coordinates(coords)

                # Array-based nested search
                elif isinstance(nested_data, (list, np.ndarray)):
                    indices = [name for name in possible_names if isinstance(name, int)]
                    nested_array = np.array(nested_data)

                    for idx in indices:
                        if nested_array.ndim >= 2 and idx < nested_array.shape[-1]:
                            if nested_array.ndim == 3:  # [frames, joints, 2]
                                return nested_array[:, idx, :2]

        return None

    def _find_array_data(self) -> Dict[str, np.ndarray]:
        """Find array-like data in the loaded data."""
        arrays = {}

        if isinstance(self.data, dict):
            for key, value in self.data.items():
                if isinstance(value, np.ndarray) and value.ndim >= 2:
                    arrays[key] = value
                elif isinstance(value, list):
                    try:
                        array_val = np.array(value)
                        if array_val.ndim >= 2:
                            arrays[key] = array_val
                    except Exception:
                        pass

        return arrays

    def _normalize_coordinates(self, coords) -> Optional[np.ndarray]:
        """Normalize coordinates to [N_frames, 2] format."""
        try:
            coords_array = np.array(coords)

            # Handle different input shapes
            if coords_array.ndim == 1:
                # Flat array, assume alternating x, y
                if len(coords_array) % 2 == 0:
                    coords_array = coords_array.reshape(-1, 2)
                else:
                    return None

            elif coords_array.ndim == 2:
                # Already in correct format or needs transpose
                if coords_array.shape[1] == 2:
                    pass  # Already correct
                elif coords_array.shape[0] == 2:
                    coords_array = coords_array.T  # Transpose
                else:
                    # Take first 2 columns
                    coords_array = coords_array[:, :2]

            elif coords_array.ndim == 3:
                # Take first 2 dimensions of last axis
                coords_array = coords_array[:, :, :2]
                if coords_array.shape[1] == 1:
                    coords_array = coords_array.squeeze(1)

            # Validate final shape
            if coords_array.ndim == 2 and coords_array.shape[1] == 2:
                return coords_array
            else:
                print(f"⚠️  Could not normalize coordinates shape: {coords_array.shape}")
                return None

        except Exception as e:
            print(f"⚠️  Error normalizing coordinates: {e}")
            return None

    def get_available_joints(self) -> List[str]:
        """Get list of available joints in the loaded data."""
        if self.data is None:
            if not self.load_ground_truth_data():
                return []

        available_joints = []

        for joint_name in self.joint_mapping.keys():
            coords = self._extract_single_joint_coordinates(joint_name)
            if coords is not None:
                available_joints.append(joint_name)

        return available_joints

    def get_data_info(self) -> Dict[str, Any]:
        """Get information about the loaded data."""
        if self.data is None:
            return {"loaded": False}

        info = {
            "loaded": True,
            "file_path": self.gt_file_path,
            "data_type": type(self.data).__name__,
        }

        if isinstance(self.data, dict):
            if "sheet_data" in self.data:
                # Excel evaluation data
                info["file_type"] = "excel_evaluation"
                info["sheets"] = list(self.data["sheet_data"].keys())
                info["primary_sheet"] = self.data["primary_sheet"]
            else:
                info["keys"] = list(self.data.keys())
                info["num_keys"] = len(self.data.keys())
        elif isinstance(self.data, pd.DataFrame):
            info["columns"] = list(self.data.columns)
            info["shape"] = self.data.shape
        elif isinstance(self.data, np.ndarray):
            info["shape"] = self.data.shape
            info["dtype"] = str(self.data.dtype)

        # Get available joints
        info["available_joints"] = self.get_available_joints()
        info["num_joints"] = len(info["available_joints"])

        return info

    def create_sample_ground_truth_file(
        self, output_path: str, num_frames: int = 100
    ) -> bool:
        """
        Create a sample ground truth file for testing.

        Args:
            output_path: Path where to save the sample file
            num_frames: Number of frames to generate

        Returns:
            True if successful, False otherwise
        """
        try:
            # Generate sample data
            sample_data = {}

            for joint_name, possible_names in self.joint_mapping.items():
                # Generate random coordinates within reasonable range
                x_coords = np.random.uniform(
                    50, 950, num_frames
                )  # Assuming 1000px width
                y_coords = np.random.uniform(
                    50, 550, num_frames
                )  # Assuming 600px height

                # Add some realistic motion (sine wave)
                t = np.linspace(0, 4 * np.pi, num_frames)
                x_coords += 20 * np.sin(t)
                y_coords += 10 * np.cos(t * 1.5)

                sample_data[joint_name] = {
                    "x": x_coords.tolist(),
                    "y": y_coords.tolist(),
                    "coordinates": np.column_stack([x_coords, y_coords]).tolist(),
                }

            # Save as JSON
            with open(output_path, "w") as f:
                json.dump(sample_data, f, indent=2)

            print(f"✅ Sample ground truth file created: {output_path}")
            return True

        except Exception as e:
            print(f"❌ Error creating sample file: {e}")
            return False


# Utility functions for backward compatibility
def load_ground_truth_coordinates(
    dataset_name: str, joint_names: List[str], gt_file_path: str = None
) -> Dict[str, np.ndarray]:
    """
    Utility function to load ground truth coordinates.

    Args:
        dataset_name: Name of the dataset
        joint_names: List of joint names to extract
        gt_file_path: Optional path to ground truth file

    Returns:
        Dictionary mapping joint names to coordinate arrays
    """
    loader = GroundTruthDataLoader(dataset_name, gt_file_path)
    return loader.extract_joint_coordinates(joint_names)


if __name__ == "__main__":
    # Example usage and testing
    print("Ground Truth Data Loader - Test Mode")
    print("=" * 50)

    # Create sample data for testing
    sample_file = "sample_ground_truth.json"
    loader = GroundTruthDataLoader("movi")

    if loader.create_sample_ground_truth_file(sample_file, num_frames=50):
        # Test loading the sample data
        loader.gt_file_path = sample_file

        joint_names = ["LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE"]
        coordinates = loader.extract_joint_coordinates(joint_names)

        print(f"\nExtracted coordinates for {len(coordinates)} joints:")
        for joint_name, coords in coordinates.items():
            print(f"  {joint_name}: shape {coords.shape}")

        # Get data info
        info = loader.get_data_info()
        print(f"\nData info: {info}")

        # Clean up
        os.remove(sample_file)
        print(f"\nCleaned up sample file: {sample_file}")

"""
Ground Truth Data Loader Script

Loads ground truth data and extracts joint coordinates.
Focus: Ground truth data loading and joint coordinate extraction only.
"""

import sys
import os
import pandas as pd
import numpy as np
from typing import List, Optional, Dict

# Add the Analysis scripts directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class GroundTruthDataLoader:
    """Load and extract ground truth joint coordinates."""

    def __init__(self, dataset_name: str):
        """Initialize with dataset name."""
        self.dataset_name = dataset_name
        self._load_config()

    def _load_config(self):
        """Load dataset configuration."""
        try:
            from config import ConfigManager

            self.config = ConfigManager.load_config(self.dataset_name)

            # Get ground truth file path
            if hasattr(self.config.paths, "ground_truth_file"):
                self.gt_file_path = self.config.paths.ground_truth_file
            else:
                raise AttributeError("No ground truth file path in config")

            # Get joint enum class
            joint_module_path = self.config.dataset.joint_enum_module
            module_parts = joint_module_path.split(".")
            module = __import__(joint_module_path, fromlist=[module_parts[-1]])
            self.joint_enum = getattr(module, module_parts[-1])

            print(f"✅ Loaded config for {self.dataset_name}")
            print(f"   Ground truth file: {self.gt_file_path}")
            print(f"   Joint enum: {self.joint_enum.__name__}")

        except Exception as e:
            print(f"❌ Failed to load config: {e}")
            raise

    def load_ground_truth_data(self) -> Optional[pd.DataFrame]:
        """Load the ground truth data file."""
        try:
            if not os.path.exists(self.gt_file_path):
                print(f"❌ Ground truth file not found: {self.gt_file_path}")
                return None

            print(f"Loading ground truth data from: {self.gt_file_path}")

            # Handle different file formats
            if self.gt_file_path.endswith(".csv"):
                data = pd.read_csv(self.gt_file_path)
            elif self.gt_file_path.endswith(".xlsx"):
                data = pd.read_excel(self.gt_file_path)
            else:
                # Try csv first
                data = pd.read_csv(self.gt_file_path)

            print(f"✅ Loaded {len(data)} records from ground truth")
            print(f"   Columns: {list(data.columns)}")

            return data

        except Exception as e:
            print(f"❌ Failed to load ground truth data: {e}")
            return None

    def get_available_joints(self) -> List[str]:
        """Get list of available joint names from the enum."""
        try:
            available_joints = [joint.name for joint in self.joint_enum]
            print(f"Available joints: {available_joints}")
            return available_joints
        except Exception as e:
            print(f"❌ Failed to get joint list: {e}")
            return []

    def get_joint_coordinate_columns(
        self, joint_names: List[str] = None
    ) -> Dict[str, List[str]]:
        """Get the column names for joint coordinates."""
        if joint_names is None:
            joint_names = self.get_available_joints()

        coord_columns = {}

        try:
            data = self.load_ground_truth_data()
            if data is None:
                return coord_columns

            # For each joint, find corresponding coordinate columns
            for joint_name in joint_names:
                joint_enum_value = getattr(self.joint_enum, joint_name)

                # Handle different joint coordinate formats
                if isinstance(joint_enum_value.value, tuple):
                    # Joint has multiple indices (e.g., for midpoint calculation)
                    x_cols = []
                    y_cols = []
                    for idx in joint_enum_value.value:
                        x_cols.append(f"x{idx}")
                        y_cols.append(f"y{idx}")
                    coord_columns[joint_name] = {
                        "x": x_cols,
                        "y": y_cols,
                        "type": "multiple",
                    }
                else:
                    # Single joint index
                    idx = joint_enum_value.value
                    x_col = f"x{idx}"
                    y_col = f"y{idx}"
                    if x_col in data.columns and y_col in data.columns:
                        coord_columns[joint_name] = {
                            "x": [x_col],
                            "y": [y_col],
                            "type": "single",
                        }
                    else:
                        print(
                            f"⚠️  Columns {x_col}, {y_col} not found for joint {joint_name}"
                        )

            print(f"✅ Found coordinate columns for {len(coord_columns)} joints")
            return coord_columns

        except Exception as e:
            print(f"❌ Failed to get coordinate columns: {e}")
            return coord_columns

    def extract_joint_coordinates(
        self,
        joint_names: List[str] = None,
        subject: str = None,
        action: str = None,
        camera: int = None,
    ) -> Dict[str, np.ndarray]:
        """Extract joint coordinates from ground truth data."""
        if joint_names is None:
            joint_names = self.get_available_joints()

        print(f"Extracting coordinates for joints: {joint_names}")

        try:
            # Load data
            data = self.load_ground_truth_data()
            if data is None:
                return {}

            # Apply filters if provided
            filtered_data = data.copy()
            if subject:
                if "Subject" in data.columns:
                    filtered_data = filtered_data[filtered_data["Subject"] == subject]
                elif "subject" in data.columns:
                    filtered_data = filtered_data[filtered_data["subject"] == subject]

            if action:
                action_cols = [col for col in data.columns if "action" in col.lower()]
                if action_cols:
                    filtered_data = filtered_data[
                        filtered_data[action_cols[0]].str.contains(action, na=False)
                    ]

            if camera is not None:
                if "Camera" in data.columns:
                    filtered_data = filtered_data[filtered_data["Camera"] == camera]
                elif "camera" in data.columns:
                    filtered_data = filtered_data[filtered_data["camera"] == camera]

            print(f"   Filtered to {len(filtered_data)} records")

            # Get coordinate columns
            coord_columns = self.get_joint_coordinate_columns(joint_names)

            joint_coordinates = {}

            for joint_name in joint_names:
                if joint_name not in coord_columns:
                    print(f"⚠️  No coordinate columns for joint {joint_name}")
                    continue

                col_info = coord_columns[joint_name]

                if col_info["type"] == "single":
                    # Single coordinate pair
                    x_col = col_info["x"][0]
                    y_col = col_info["y"][0]

                    if (
                        x_col in filtered_data.columns
                        and y_col in filtered_data.columns
                    ):
                        coords = filtered_data[[x_col, y_col]].values
                        joint_coordinates[joint_name] = coords
                        print(f"   ✅ {joint_name}: {len(coords)} coordinate pairs")
                    else:
                        print(f"   ⚠️  Missing columns for {joint_name}")

                elif col_info["type"] == "multiple":
                    # Multiple coordinates - calculate midpoint
                    x_coords = []
                    y_coords = []

                    for x_col, y_col in zip(col_info["x"], col_info["y"]):
                        if (
                            x_col in filtered_data.columns
                            and y_col in filtered_data.columns
                        ):
                            x_coords.append(filtered_data[x_col].values)
                            y_coords.append(filtered_data[y_col].values)

                    if x_coords and y_coords:
                        # Calculate midpoint
                        x_mid = np.mean(x_coords, axis=0)
                        y_mid = np.mean(y_coords, axis=0)
                        coords = np.column_stack([x_mid, y_mid])
                        joint_coordinates[joint_name] = coords
                        print(
                            f"   ✅ {joint_name} (midpoint): {len(coords)} coordinate pairs"
                        )
                    else:
                        print(f"   ⚠️  Missing columns for {joint_name}")

            return joint_coordinates

        except Exception as e:
            print(f"❌ Failed to extract joint coordinates: {e}")
            return {}

    def get_data_summary(self) -> Dict:
        """Get summary information about the ground truth data."""
        try:
            data = self.load_ground_truth_data()
            if data is None:
                return {}

            summary = {
                "total_records": len(data),
                "columns": list(data.columns),
                "available_joints": self.get_available_joints(),
                "coordinate_columns": self.get_joint_coordinate_columns(),
            }

            # Check for common metadata columns
            for col in [
                "Subject",
                "subject",
                "Action",
                "action",
                "Camera",
                "camera",
                "Frame",
                "frame",
            ]:
                if col in data.columns:
                    unique_values = data[col].nunique()
                    sample_values = data[col].unique()[:5].tolist()
                    summary[f"{col}_info"] = {
                        "unique_count": unique_values,
                        "sample_values": sample_values,
                    }

            return summary

        except Exception as e:
            print(f"❌ Failed to get data summary: {e}")
            return {}

    def export_joint_coordinates_to_csv(
        self,
        joint_names: List[str] = None,
        output_filename: str = None,
        **filter_kwargs,
    ) -> str:
        """Export joint coordinates to CSV file."""
        try:
            coordinates = self.extract_joint_coordinates(joint_names, **filter_kwargs)

            if not coordinates:
                print("❌ No coordinates to export")
                return ""

            # Prepare data for CSV
            export_data = []
            max_length = max(len(coords) for coords in coordinates.values())

            for i in range(max_length):
                row = {"frame": i}
                for joint_name, coords in coordinates.items():
                    if i < len(coords):
                        row[f"{joint_name}_x"] = coords[i][0]
                        row[f"{joint_name}_y"] = coords[i][1]
                    else:
                        row[f"{joint_name}_x"] = np.nan
                        row[f"{joint_name}_y"] = np.nan
                export_data.append(row)

            export_df = pd.DataFrame(export_data)

            # Generate filename
            if output_filename is None:
                joint_str = "_".join(joint_names[:3]) if joint_names else "all_joints"
                output_filename = f"gt_coordinates_{self.dataset_name}_{joint_str}.csv"

            # Save to config save folder
            output_path = os.path.join(self.config.save_folder, output_filename)
            os.makedirs(self.config.save_folder, exist_ok=True)

            export_df.to_csv(output_path, index=False)
            print(f"✅ Exported joint coordinates to: {output_path}")

            return output_path

        except Exception as e:
            print(f"❌ Failed to export coordinates: {e}")
            return ""


def main():
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Ground Truth Data Loader")
    parser.add_argument("dataset", help="Dataset name (e.g., 'humaneva', 'movi')")
    parser.add_argument(
        "--joints", nargs="*", help="Specific joints to extract (default: all)"
    )
    parser.add_argument("--subject", help="Filter by subject")
    parser.add_argument("--action", help="Filter by action")
    parser.add_argument("--camera", type=int, help="Filter by camera")
    parser.add_argument("--summary", action="store_true", help="Show data summary")
    parser.add_argument(
        "--export", action="store_true", help="Export coordinates to CSV"
    )
    parser.add_argument("--filename", help="Custom output filename")

    args = parser.parse_args()

    try:
        loader = GroundTruthDataLoader(args.dataset)

        if args.summary:
            summary = loader.get_data_summary()
            print("\n" + "=" * 50)
            print("GROUND TRUTH DATA SUMMARY")
            print("=" * 50)
            for key, value in summary.items():
                print(f"{key}: {value}")
            return

        # Extract coordinates
        filter_kwargs = {}
        if args.subject:
            filter_kwargs["subject"] = args.subject
        if args.action:
            filter_kwargs["action"] = args.action
        if args.camera is not None:
            filter_kwargs["camera"] = args.camera

        coordinates = loader.extract_joint_coordinates(args.joints, **filter_kwargs)

        if coordinates:
            print(
                f"\n✅ Successfully extracted coordinates for {len(coordinates)} joints"
            )
            for joint_name, coords in coordinates.items():
                print(f"   {joint_name}: {len(coords)} frames")

        if args.export:
            loader.export_joint_coordinates_to_csv(
                args.joints, args.filename, **filter_kwargs
            )

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

"""
Config Extractor

Handles extraction of configuration data for different analysis types.
"""

from typing import Optional, List, Tuple
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config import load_dataset_analysis_config


def extract_joint_analysis_config(
    dataset_name: str,
) -> Tuple[Optional[List[str]], Optional[List[float]]]:
    """Extract joints and PCK thresholds from dataset config.

    Args:
        dataset_name: Name of the dataset

    Returns:
        Tuple[joints_to_analyze, pck_thresholds]: Extracted config data or None if not found
    """
    try:
        analysis_config = load_dataset_analysis_config(dataset_name)

        joints_to_analyze = None
        pck_thresholds = None

        if hasattr(analysis_config, "config_dict") and analysis_config.config_dict:
            pck_scores_config = analysis_config.config_dict.get("pck_scores", {})
            jointwise_columns = pck_scores_config.get("jointwise", [])

            # Extract unique thresholds and joints from jointwise column names
            # Column names are like: 'LEFT_HIP_jointwise_pck_0.01'
            thresholds_set = set()
            joints_set = set()

            for col in jointwise_columns:
                if "_jointwise_pck_" in col:
                    # Extract threshold
                    threshold_str = col.split("_pck_")[-1]
                    try:
                        threshold = float(threshold_str)
                        thresholds_set.add(threshold)
                    except ValueError:
                        continue

                    # Extract joint name (everything before '_jointwise_pck_')
                    joint_name = col.split("_jointwise_pck_")[0]
                    if joint_name:
                        joints_set.add(joint_name)

            if thresholds_set:
                pck_thresholds = sorted(list(thresholds_set))
                print(f"Extracted PCK thresholds from config: {pck_thresholds}")
            else:
                print("No PCK thresholds found in config, will use defaults")

            if joints_set:
                joints_to_analyze = sorted(list(joints_set))
                print(f"Extracted joints from config: {joints_to_analyze}")
            else:
                print("No joints found in config, will use defaults")

        return joints_to_analyze, pck_thresholds

    except Exception as e:
        print(f"WARNING: Failed to extract joint analysis config: {e}")
        return None, None


def extract_analysis_paths(dataset_name: str) -> dict:
    """Extract file paths from dataset config.

    Args:
        dataset_name: Name of the dataset

    Returns:
        dict: Dictionary containing extracted paths
    """
    try:
        analysis_config = load_dataset_analysis_config(dataset_name)

        paths = {}

        if hasattr(analysis_config, "config_dict") and analysis_config.config_dict:
            paths_config = analysis_config.config_dict.get("paths", {})

            paths = {
                "video_directory": paths_config.get("video_directory"),
                "pck_file_path": paths_config.get("pck_file_path"),
                "save_folder": paths_config.get("save_folder"),
                "ground_truth_file": paths_config.get("ground_truth_file"),
            }

            print(f"Extracted paths from config: {list(paths.keys())}")

        return paths

    except Exception as e:
        print(f"WARNING: Failed to extract analysis paths: {e}")
        return {}


def extract_dataset_info(dataset_name: str) -> dict:
    """Extract dataset information from config.

    Args:
        dataset_name: Name of the dataset

    Returns:
        dict: Dictionary containing dataset information
    """
    try:
        analysis_config = load_dataset_analysis_config(dataset_name)

        dataset_info = {}

        if hasattr(analysis_config, "config_dict") and analysis_config.config_dict:
            dataset_config = analysis_config.config_dict.get("dataset", {})

            dataset_info = {
                "name": dataset_config.get("name", dataset_name),
                "model": dataset_config.get("model"),
            }

            # Extract column configuration
            columns_config = analysis_config.config_dict.get("columns", {})
            dataset_info["columns"] = {
                "subject_column": columns_config.get("subject_column"),
                "action_column": columns_config.get("action_column"),
                "camera_column": columns_config.get("camera_column"),
            }

            print(f"Extracted dataset info: {dataset_info}")

        return dataset_info

    except Exception as e:
        print(f"WARNING: Failed to extract dataset info: {e}")
        return {"name": dataset_name}


def get_analysis_settings(dataset_name: str) -> dict:
    """Get analysis-specific settings from config.

    Args:
        dataset_name: Name of the dataset

    Returns:
        dict: Dictionary containing analysis settings
    """
    try:
        analysis_config = load_dataset_analysis_config(dataset_name)

        settings = {
            "save_results": True,
            "create_plots": True,
            "generate_reports": True,
        }

        if hasattr(analysis_config, "config_dict") and analysis_config.config_dict:
            analysis_settings = analysis_config.config_dict.get("analysis", {})

            # Extract joint brightness settings
            joint_brightness = analysis_settings.get("joint_brightness", {})
            if joint_brightness:
                settings["sampling_radius"] = joint_brightness.get("sampling_radius", 3)
                settings["brightness_settings"] = joint_brightness.get(
                    "brightness_settings", {}
                )

            # Extract visualization settings
            pck_brightness = analysis_settings.get("pck_brightness", {})
            if pck_brightness:
                settings["visualization"] = pck_brightness.get("visualization", {})
                settings["export"] = pck_brightness.get("export", {})

        return settings

    except Exception as e:
        print(f"WARNING: Failed to extract analysis settings: {e}")
        return {
            "save_results": True,
            "create_plots": True,
            "generate_reports": True,
        }

"""
Factory for creating dataset-specific configurations.
"""

import yaml
import os
from types import SimpleNamespace
from .dataset_config import DatasetConfig


class ConfigFactory:
    """Factory for creating dataset configurations."""

    @staticmethod
    def create_config(dataset_name: str) -> DatasetConfig:
        """Create configuration for a specific dataset from YAML."""
        dataset_name = dataset_name.lower()

        if dataset_name in ("humaneva", "movi"):
            return ConfigFactory._create_config_from_yaml(dataset_name)
        else:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. "
                f"Supported datasets are 'humaneva' and 'movi'."
            )

    @staticmethod
    def _create_config_from_yaml(dataset_name: str) -> DatasetConfig:
        """Create configuration from YAML file."""
        # Get the path to the YAML config file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        analysis_scripts_root = os.path.dirname(current_dir)
        config_path = os.path.join(
            analysis_scripts_root, "config_yamls", f"{dataset_name}_config.yaml"
        )

        try:
            with open(config_path, "r") as file:
                config_data = yaml.safe_load(file)

            # Extract configuration values
            dataset_info = config_data.get("dataset", {})
            paths = config_data.get("paths", {})
            columns = config_data.get("columns", {})
            pck_scores = config_data.get("pck_scores", {})
            sync_data_raw = config_data.get("sync_data")
            analysis_config = config_data.get("analysis", {})

            # Convert sync_data to SimpleNamespace if it exists
            sync_data = None
            if sync_data_raw is not None:
                sync_data = SimpleNamespace(data=sync_data_raw)

            # Create DatasetConfig instance
            return DatasetConfig(
                name=dataset_info.get("name", dataset_name),
                video_directory=paths.get("video_directory", ""),
                pck_file_path=paths.get("pck_file_path", ""),
                save_folder=paths.get("save_folder", ""),
                model=dataset_info.get("model", ""),
                subject_column=columns.get("subject_column"),
                action_column=columns.get("action_column"),
                camera_column=columns.get("camera_column"),
                pck_overall_score_columns=pck_scores.get("overall", []),
                pck_per_frame_score_columns=pck_scores.get("per_frame", []),
                sync_data=sync_data,
                analysis_config=analysis_config,  # Pass analysis config directly
            )

        except FileNotFoundError:
            raise ValueError(f"Configuration file not found: {config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML configuration: {e}")
        except Exception as e:
            raise ValueError(f"Error loading configuration for {dataset_name}: {e}")

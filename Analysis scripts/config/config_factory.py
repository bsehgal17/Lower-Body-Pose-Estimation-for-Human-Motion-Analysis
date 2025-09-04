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
        """Create configuration for a specific dataset."""
        dataset_name = dataset_name.lower()

        if dataset_name == "humaneva":
            return ConfigFactory._create_config_from_yaml("humaneva")
        elif dataset_name == "movi":
            return ConfigFactory._create_config_from_yaml("movi")
        else:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. Supported datasets are 'humaneva' and 'movi'."
            )

    @staticmethod
    def _create_config_from_yaml(dataset_name: str) -> DatasetConfig:
        """Create configuration from YAML file."""
        # Get the path to the YAML config file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        config_path = os.path.join(
            project_root, "config_yamls", f"{dataset_name}_config.yaml"
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
            dataset_config = DatasetConfig(
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
            )

            # Add analysis config as an attribute if it exists
            if analysis_config:
                dataset_config.analysis_config = analysis_config

            return dataset_config

        except FileNotFoundError:
            raise ValueError(f"Configuration file not found: {config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML configuration: {e}")
        except Exception as e:
            raise ValueError(f"Error loading configuration for {dataset_name}: {e}")

    @staticmethod
    def _create_humaneva_config() -> DatasetConfig:
        """Create HumanEva dataset configuration (DEPRECATED - use YAML)."""
        print(
            "Warning: Using deprecated hardcoded configuration. Please use YAML configuration instead."
        )
        from types import SimpleNamespace

        sync_data = SimpleNamespace(
            data={
                "S1": {"Walking 1": [667, 667, 667], "Jog 1": [49, 50, 51]},
                "S2": {"Walking 1": [547, 547, 546], "Jog 1": [493, 491, 502]},
                "S3": {"Walking 1": [524, 524, 524], "Jog 1": [464, 462, 462]},
            }
        )

        return DatasetConfig(
            name="humaneva",
            video_directory="/storage/Projects/Gaitly/bsehgal/lower_body_pose_est/HumanEva",
            pck_file_path="/storage/Projects/Gaitly/bsehgal/lower_body_pose_est/pipeline_results/HumanEva/detect_RTMW_X/evaluation/2025-08-21_17-21-05/detect/detect_metrics.xlsx",
            save_folder="/storage/Projects/Gaitly/bsehgal/lower_body_pose_est/pipeline_results/analysis_results/HumanEva/High_threshold",
            model="RTMW",
            subject_column="subject",
            action_column="action",
            camera_column="camera",
            pck_overall_score_columns=[
                "overall_overall_pck_0.10",
                "overall_overall_pck_0.20",
                "overall_overall_pck_0.50",
            ],
            pck_per_frame_score_columns=[
                "pck_per_frame_pck_0.10",
                "pck_per_frame_pck_0.20",
                "pck_per_frame_pck_0.50",
            ],
            sync_data=sync_data,
        )

    @staticmethod
    def _create_movi_config() -> DatasetConfig:
        """Create MoVi dataset configuration (DEPRECATED - use YAML)."""
        print(
            "Warning: Using deprecated hardcoded configuration. Please use YAML configuration instead."
        )
        return DatasetConfig(
            name="movi",
            video_directory="/storage/Projects/Gaitly/bsehgal/lower_body_pose_est/MoVi",
            pck_file_path="/storage/Projects/Gaitly/bsehgal/lower_body_pose_est/pipeline_results/MoVi/detect_RTMW/evaluation/2025-08-22_17-37-59/2025-08-22_17-37-59_metrics.xlsx",
            save_folder="/storage/Projects/Gaitly/bsehgal/lower_body_pose_est/pipeline_results/analysis_results/MoVi/Testing",
            model="RTMW",
            subject_column="subject",
            action_column=None,
            camera_column=None,
            pck_overall_score_columns=[
                "overall_overall_pck_0.25",
                "overall_overall_pck_0.30",
            ],
            pck_per_frame_score_columns=[
                "pck_per_frame_pck_0.25",
                "pck_per_frame_pck_0.30",
            ],
            sync_data=None,
        )

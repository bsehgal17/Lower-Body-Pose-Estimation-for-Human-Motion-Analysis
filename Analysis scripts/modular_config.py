"""
Enhanced configuration management system.
"""

import os
from typing import Any, List
from dataclasses import dataclass


@dataclass
class DatasetConfig:
    """Base configuration for datasets."""

    name: str
    video_directory: str
    pck_file_path: str
    save_folder: str
    model: str
    subject_column: str
    action_column: str
    camera_column: str
    pck_overall_score_columns: List[str]
    pck_per_frame_score_columns: List[str]
    sync_data: Any

    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()

    def validate(self) -> bool:
        """Validate the configuration."""
        errors = []

        if not os.path.exists(self.video_directory):
            errors.append(f"Video directory does not exist: {self.video_directory}")

        if not os.path.exists(self.pck_file_path):
            errors.append(f"PCK file does not exist: {self.pck_file_path}")

        if not self.pck_overall_score_columns:
            errors.append("PCK overall score columns not specified")

        if not self.pck_per_frame_score_columns:
            errors.append("PCK per-frame score columns not specified")

        if errors:
            for error in errors:
                print(f"Configuration Error: {error}")
            return False

        return True

    def get_grouping_columns(self) -> List[str]:
        """Get the columns used for grouping video data."""
        return [
            col
            for col in [self.subject_column, self.action_column, self.camera_column]
            if col is not None
        ]


class ConfigFactory:
    """Factory for creating dataset configurations."""

    @staticmethod
    def create_config(dataset_name: str) -> DatasetConfig:
        """Create configuration for a specific dataset."""
        dataset_name = dataset_name.lower()

        if dataset_name == "humaneva":
            return ConfigFactory._create_humaneva_config()
        elif dataset_name == "movi":
            return ConfigFactory._create_movi_config()
        else:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. Supported datasets are 'humaneva' and 'movi'."
            )

    @staticmethod
    def _create_humaneva_config() -> DatasetConfig:
        """Create HumanEva dataset configuration."""
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
        """Create MoVi dataset configuration."""
        return DatasetConfig(
            name="movi",
            video_directory="/storage/Projects/Gaitly/bsehgal/lower_body_pose_est/MoVi",
            pck_file_path="/storage/Projects/Gaitly/bsehgal/lower_body_pose_est/pipeline_results/MoVi/detect_RTMW/evaluation/2025-08-22_17-37-59/2025-08-22_17-37-59_metrics.xlsx",
            save_folder="/storage/Projects/Gaitly/bsehgal/lower_body_pose_est/pipeline_results/analysis_results/MoVi/High_threshold",
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


class ConfigManager:
    """Enhanced configuration manager."""

    @staticmethod
    def load_config(dataset_name: str) -> DatasetConfig:
        """Load configuration for a dataset."""
        try:
            config = ConfigFactory.create_config(dataset_name)
            print(f"Configuration loaded for {dataset_name.upper()} dataset")
            return config
        except ValueError as e:
            print(f"Error loading configuration: {e}")
            raise

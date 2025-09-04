"""
Dataset configuration definitions.
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
    pck_jointwise_score_columns: List[str] = None
    sync_data: Any
    analysis_config: Any = None  # Analysis-specific configuration
    ground_truth_file: str = None  # Path to ground truth coordinates file

    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()

    def get_analysis_bin_size(
        self, analysis_type: str = "pck_brightness", default: int = 5
    ) -> int:
        """Get bin size for analysis from configuration."""
        if not self.analysis_config:
            return default

        analysis_section = self.analysis_config.get(analysis_type, {})
        return analysis_section.get("bin_size", default)

    def get_analysis_score_groups(self, analysis_type: str = "pck_brightness") -> dict:
        """Get score groups for analysis from configuration."""
        if not self.analysis_config:
            return {}

        analysis_section = self.analysis_config.get(analysis_type, {})
        return analysis_section.get("score_groups", {})

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

        # Make jointwise columns optional for backward compatibility
        if self.pck_jointwise_score_columns is None:
            self.pck_jointwise_score_columns = []
            print(
                "Warning: PCK jointwise score columns not specified, using empty list"
            )
        elif not self.pck_jointwise_score_columns:
            print("Warning: PCK jointwise score columns list is empty")

        # Optional validation for ground truth file
        if self.ground_truth_file and not os.path.exists(self.ground_truth_file):
            errors.append(f"Ground truth file does not exist: {self.ground_truth_file}")

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

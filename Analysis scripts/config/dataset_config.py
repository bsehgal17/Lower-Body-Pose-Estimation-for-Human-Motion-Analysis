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
    sync_data: Any = None
    analysis_config: Any = None  # Analysis-specific configuration
    ground_truth_file: str = None  # Path to ground truth coordinates file
    grouping_columns: List[str] = None  # Columns to use for video grouping
    video_name_format: str = "{subject}"  # Format for creating video names

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
        # Use the explicitly defined grouping columns if available
        if self.grouping_columns:
            return self.grouping_columns

        # Fallback to the old logic for backward compatibility
        return [
            col
            for col in [self.subject_column, self.action_column, self.camera_column]
            if col is not None
        ]

    def create_video_name(self, group_key, grouping_cols: List[str]) -> str:
        """Create a video name using the configured format.

        Args:
            group_key: The values for the grouping columns (single value or tuple)
            grouping_cols: List of column names used for grouping

        Returns:
            Formatted video name string
        """
        try:
            # If single grouping column, group_key is a single value
            if len(grouping_cols) == 1:
                format_dict = {grouping_cols[0]: str(group_key)}
            else:
                # Multiple grouping columns - group_key is a tuple
                format_dict = {}
                for i, col in enumerate(grouping_cols):
                    if i < len(group_key):
                        format_dict[col] = str(group_key[i])
                    else:
                        format_dict[col] = "Unknown"

            # Use the configured format string
            return self.video_name_format.format(**format_dict)

        except (KeyError, IndexError, AttributeError) as e:
            # Fallback to simple string representation if formatting fails
            print(
                f"Warning: Failed to format video name with {self.video_name_format}: {e}"
            )
            if len(grouping_cols) == 1:
                return str(group_key)
            else:
                return "_".join(str(val) for val in group_key)

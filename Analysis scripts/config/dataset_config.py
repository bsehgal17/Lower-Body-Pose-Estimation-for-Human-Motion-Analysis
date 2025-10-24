"""
Dataset configuration definitions.
"""

import os
from typing import Any, List, Optional
from pydantic import BaseModel, model_validator, Field


class DatasetConfig(BaseModel):
    """
    Configuration for dataset analysis and evaluation parameters.

    Defines dataset-specific paths, file formats, evaluation metrics,
    and analysis parameters for pose estimation performance assessment.
    """

    name: str = Field(
        ...,
        description="Human-readable name identifier for the dataset. "
        "Examples: 'HumanEva-I', 'MPI-INF-3DHP', 'COCO', 'MoVi'. "
        "Used in reports and output file naming.",
    )

    video_directory: str = Field(
        ...,
        description="Root directory path containing all video files for this dataset. "
        "Should contain subdirectories organized by subject/action/camera "
        "or flat structure depending on dataset organization.",
    )

    pck_file_path: str = Field(
        ...,
        description="Path to CSV/Excel file containing PCK (Percentage of Correct Keypoints) "
        "evaluation results. Must contain columns specified in pck_*_columns fields. "
        "This is the main evaluation data source for analysis.",
    )

    save_folder: str = Field(
        ...,
        description="Output directory where analysis results, plots, and reports will be saved. "
        "Directory structure will be created automatically if it doesn't exist.",
    )

    model: str = Field(
        ...,
        description="Pose estimation model identifier used to generate the PCK results. "
        "Examples: 'RTMPose', 'HRNet', 'PoseNet', 'MediaPipe'. "
        "Used for labeling and organizing results by model type.",
    )

    subject_column: str = Field(
        ...,
        description="Column name in PCK file that identifies different subjects/persons. "
        "Examples: 'Subject', 'Person_ID', 'subject_id'. "
        "Used for subject-wise analysis and grouping.",
    )

    action_column: Optional[str] = Field(
        default=None,
        description="(Optional) Column name in PCK file identifying actions/activities. "
        "Examples: 'Action', 'Activity', 'action_name'. "
        "Used for action-wise performance analysis.",
    )

    camera_column: Optional[str] = Field(
        default=None,
        description="(Optional) Column name in PCK file identifying camera views. "
        "Examples: 'Camera', 'View', 'camera_id'. "
        "Used for camera/viewpoint-specific analysis.",
    )

    pck_overall_score_columns: List[str] = Field(
        ...,
        description="List of column names containing overall PCK scores (0.0-1.0). "
        "These represent aggregate pose accuracy across all keypoints. "
        "Examples: ['PCK_0.2', 'PCK_0.5'] for different distance thresholds.",
    )

    pck_per_frame_score_columns: List[str] = Field(
        ...,
        description="List of column names containing per-frame PCK scores. "
        "These represent pose accuracy for individual video frames. "
        "Used for temporal analysis and frame-level statistics.",
    )

    pck_jointwise_score_columns: Optional[List[str]] = Field(
        default=None,
        description="List of column names containing joint-specific PCK scores. "
        "Examples: ['PCK_nose', 'PCK_left_shoulder', 'PCK_right_knee']. "
        "Used for analyzing which body parts are estimated most accurately.",
    )

    sync_data: Optional[Any] = Field(
        default=None,
        description="Synchronization information for multi-camera datasets. "
        "Format depends on dataset structure. Used for temporal alignment "
        "across different camera views or sequences.",
    )

    analysis_config: Optional[Any] = Field(
        default=None,
        description="Analysis-specific configuration parameters. "
        "May contain binning parameters, score grouping thresholds, "
        "statistical test settings, or custom analysis options.",
    )

    ground_truth_file: Optional[str] = Field(
        default=None,
        description="Path to ground truth keypoint coordinate file. "
        "Used for direct coordinate comparison and error analysis "
        "beyond PCK metrics. Format typically CSV or NPY.",
    )

    grouping_columns: Optional[List[str]] = Field(
        default=None,
        description="List of column names to use for grouping analysis results. "
        "If None, defaults to [subject_column, action_column, camera_column]. "
        "Determines how data is aggregated for statistical analysis.",
    )

    video_name_format: str = Field(
        default="{subject}",
        description="Format string for generating video file names from data columns. "
        "Supports column name substitution and arithmetic: '{subject}', "
        "'{action}_{camera}', 'C{camera+1}'. Used to match PCK data "
        "with corresponding video files.",
    )

    @model_validator(mode="after")
    def validate_config(self):
        """Validate configuration after initialization."""
        self.validate()
        return self

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
            errors.append(
                f"Video directory does not exist: {self.video_directory}")

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
            errors.append(
                f"Ground truth file does not exist: {self.ground_truth_file}")

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
                format_dict = {grouping_cols[0]: group_key}
            else:
                # Multiple grouping columns - group_key is a tuple
                format_dict = {}
                for i, col in enumerate(grouping_cols):
                    if i < len(group_key):
                        format_dict[col] = group_key[i]
                    else:
                        format_dict[col] = "Unknown"

            # Handle special formatting cases
            formatted_name = self._apply_custom_formatting(
                self.video_name_format, format_dict
            )
            return formatted_name

        except (KeyError, IndexError, AttributeError) as e:
            # Fallback to simple string representation if formatting fails
            print(
                f"Warning: Failed to format video name with {self.video_name_format}: {e}"
            )
            if len(grouping_cols) == 1:
                return str(group_key)
            else:
                return "_".join(str(val) for val in group_key)

    def _apply_custom_formatting(self, format_string: str, format_dict: dict) -> str:
        """Apply custom formatting that supports arithmetic and type conversions.

        Handles cases like:
        - {camera+1} - arithmetic on integer values
        - C{camera+1} - prefix/suffix with arithmetic
        """
        import re

        # Handle arithmetic expressions like {camera+1}, {camera-1}, etc.
        def replace_arithmetic(match):
            expression = match.group(1)

            # Check for arithmetic operations
            if "+" in expression:
                var_name, add_value = expression.split("+", 1)
                var_name = var_name.strip()
                add_value = add_value.strip()
                try:
                    base_value = format_dict.get(var_name, 0)
                    # Convert to int if it's a string representation of a number
                    if isinstance(base_value, str) and base_value.isdigit():
                        base_value = int(base_value)
                    elif isinstance(base_value, str):
                        # Try to extract number from string like "C1" -> 1
                        number_match = re.search(r"\d+", base_value)
                        if number_match:
                            base_value = int(number_match.group())
                        else:
                            base_value = 0

                    result = int(base_value) + int(add_value)
                    return str(result)
                except (ValueError, TypeError):
                    return expression  # Return original if can't process

            elif "-" in expression:
                var_name, sub_value = expression.split("-", 1)
                var_name = var_name.strip()
                sub_value = sub_value.strip()
                try:
                    base_value = format_dict.get(var_name, 0)
                    # Convert to int if it's a string representation of a number
                    if isinstance(base_value, str) and base_value.isdigit():
                        base_value = int(base_value)
                    elif isinstance(base_value, str):
                        # Try to extract number from string like "C1" -> 1
                        number_match = re.search(r"\d+", base_value)
                        if number_match:
                            base_value = int(number_match.group())
                        else:
                            base_value = 0

                    result = int(base_value) - int(sub_value)
                    return str(result)
                except (ValueError, TypeError):
                    return expression  # Return original if can't process
            else:
                # No arithmetic, just return the variable value
                var_value = format_dict.get(expression, expression)
                return str(var_value)

        # Find and replace arithmetic expressions
        arithmetic_pattern = r"\{([^}]+[\+\-][^}]+)\}"
        processed_string = re.sub(
            arithmetic_pattern, replace_arithmetic, format_string)

        # Now handle regular formatting
        try:
            return processed_string.format(**format_dict)
        except KeyError:
            # If some variables are still missing, do a safer replacement
            for key, value in format_dict.items():
                processed_string = processed_string.replace(
                    f"{{{key}}}", str(value))
            return processed_string

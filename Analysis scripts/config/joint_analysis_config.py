"""
Joint Analysis Configuration

Configuration settings for joint analysis.
"""

from typing import List


class JointAnalysisConfig:
    """Configuration class for joint analysis settings."""

    # Default joints to analyze (hip, knee, ankle - both left and right)
    DEFAULT_JOINTS = [
        "LEFT_HIP",
        "RIGHT_HIP",
        "LEFT_KNEE",
        "RIGHT_KNEE",
        "LEFT_ANKLE",
        "RIGHT_ANKLE",
    ]

    # Default PCK thresholds for analysis
    DEFAULT_PCK_THRESHOLDS = [0.01, 0.05, 0.1]

    # Default datasets
    AVAILABLE_DATASETS = ["movi", "humaneva"]

    @classmethod
    def get_default_config(cls, dataset_name: str = "movi") -> dict:
        """Get default configuration for joint analysis.

        Args:
            dataset_name: Name of the dataset

        Returns:
            dict: Default configuration
        """
        return {
            "dataset_name": dataset_name,
            "joints_to_analyze": cls.DEFAULT_JOINTS.copy(),
            "pck_thresholds": cls.DEFAULT_PCK_THRESHOLDS.copy(),
            "save_results": True,
            "output_dir": None,  # Auto-generated
        }

    @classmethod
    def validate_joints(cls, joints: List[str]) -> bool:
        """Validate that joints are in the expected format.

        Args:
            joints: List of joint names to validate

        Returns:
            bool: True if joints are valid
        """
        valid_joint_parts = [
            "LEFT_HIP",
            "RIGHT_HIP",
            "LEFT_KNEE",
            "RIGHT_KNEE",
            "LEFT_ANKLE",
            "RIGHT_ANKLE",
        ]

        for joint in joints:
            if joint not in valid_joint_parts:
                print(
                    f"WARNING: Unknown joint '{joint}'. Expected one of: {valid_joint_parts}"
                )
                return False

        return True

    @classmethod
    def validate_thresholds(cls, thresholds: List[float]) -> bool:
        """Validate PCK thresholds.

        Args:
            thresholds: List of threshold values

        Returns:
            bool: True if thresholds are valid
        """
        for threshold in thresholds:
            if not (0 < threshold <= 1.0):
                print(f"WARNING: Threshold {threshold} should be between 0 and 1.0")
                return False

        return True

    @classmethod
    def validate_dataset(cls, dataset_name: str) -> bool:
        """Validate dataset name.

        Args:
            dataset_name: Name of the dataset

        Returns:
            bool: True if dataset is valid
        """
        if dataset_name not in cls.AVAILABLE_DATASETS:
            print(
                f"WARNING: Unknown dataset '{dataset_name}'. Available: {cls.AVAILABLE_DATASETS}"
            )
            return False

        return True

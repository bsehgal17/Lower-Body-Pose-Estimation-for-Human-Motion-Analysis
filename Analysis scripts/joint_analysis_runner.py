"""
Joint Analysis Main Runner

Simple main entry point for running joint analysis from main.py or standalone.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from core.joint_analysis_pipeline import JointAnalysisPipeline
from config.joint_analysis_config import JointAnalysisConfig


def run_joint_analysis(
    dataset_name: str = "movi",
    joints_to_analyze: list = None,
    pck_thresholds: list = None,
    output_dir: str = None,
    save_results: bool = True,
) -> bool:
    """Run joint analysis with specified parameters.

    Args:
        dataset_name: Name of the dataset to analyze
        joints_to_analyze: List of joints to analyze (uses default if None)
        pck_thresholds: List of PCK thresholds (uses default if None)
        output_dir: Output directory (auto-generated if None)
        save_results: Whether to save results to files

    Returns:
        bool: True if analysis completed successfully
    """
    try:
        # Use defaults if not specified
        if joints_to_analyze is None:
            joints_to_analyze = JointAnalysisConfig.DEFAULT_JOINTS

        if pck_thresholds is None:
            pck_thresholds = JointAnalysisConfig.DEFAULT_PCK_THRESHOLDS

        # Validate inputs
        if not JointAnalysisConfig.validate_dataset(dataset_name):
            return False

        if not JointAnalysisConfig.validate_joints(joints_to_analyze):
            return False

        if not JointAnalysisConfig.validate_thresholds(pck_thresholds):
            return False

        # Create and run pipeline
        pipeline = JointAnalysisPipeline(
            dataset_name=dataset_name,
            joints_to_analyze=joints_to_analyze,
            pck_thresholds=pck_thresholds,
            output_dir=output_dir,
            save_results=save_results,
        )

        return pipeline.run_complete_analysis()

    except Exception as e:
        print(f"ERROR: Joint analysis failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def run_quick_analysis(dataset_name: str = "movi") -> bool:
    """Run joint analysis with default settings.

    Args:
        dataset_name: Name of the dataset to analyze

    Returns:
        bool: True if analysis completed successfully
    """
    print("Running Quick Joint Analysis")
    print("-" * 30)

    return run_joint_analysis(
        dataset_name=dataset_name,
        joints_to_analyze=None,  # Use defaults
        pck_thresholds=None,  # Use defaults
        output_dir=None,  # Auto-generate
        save_results=True,
    )

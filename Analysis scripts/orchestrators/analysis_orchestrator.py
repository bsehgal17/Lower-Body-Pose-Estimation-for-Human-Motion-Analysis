"""
Analysis Orchestrator

Main coordinator for different types of analysis pipelines.
"""

import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config import load_dataset_analysis_config
from utils.config_extractor import get_analysis_settings
from runners.single_analysis_runner import run_single_analysis
from per_video_joint_analysis_runner import run_joint_level_analysis


class AnalysisOrchestrator:
    """Orchestrates different types of analysis pipelines."""

    def __init__(self, dataset_name: str):
        """Initialize the orchestrator.

        Args:
            dataset_name: Name of the dataset to analyze
        """
        self.dataset_name = dataset_name
        self.analysis_config = None
        self.settings = None

        # Load configuration
        self._load_configuration()

    def _load_configuration(self):
        """Load analysis configuration and settings."""
        try:
            self.analysis_config = load_dataset_analysis_config(self.dataset_name)
            self.settings = get_analysis_settings(self.dataset_name)
            print(f"Configuration loaded for dataset: {self.dataset_name}")
        except Exception as e:
            print(f"WARNING: Failed to load configuration: {e}")
            self.settings = {"save_results": True, "create_plots": True}

    def run_standard_analysis(self, metrics_config: Optional[dict] = None) -> bool:
        """Run standard single analysis pipeline.

        Args:
            metrics_config: Configuration for metrics

        Returns:
            bool: True if analysis completed successfully
        """
        print("=== STANDARD ANALYSIS PIPELINE ===")

        if metrics_config is None:
            metrics_config = {"brightness": "get_brightness_data"}

        return run_single_analysis(
            dataset_name=self.dataset_name,
            metrics_config=metrics_config,
            analysis_config=self.analysis_config,
        )

    def run_joint_level_analysis(
        self,
        custom_joints: Optional[list] = None,
        output_dir: Optional[str] = None,
        sampling_radius: int = 3,
    ) -> bool:
        """Run joint-level brightness analysis pipeline.

        Args:
            custom_joints: Custom list of joints (uses default if None)
            output_dir: Custom output directory
            sampling_radius: Radius for brightness sampling around joints

        Returns:
            bool: True if analysis completed successfully
        """
        print("=== JOINT-LEVEL BRIGHTNESS ANALYSIS ===")

        # Use default joints if not provided
        if custom_joints is None:
            # Default lower body joints
            joints_to_analyze = [
                "LEFT_HIP",
                "RIGHT_HIP",
                "LEFT_KNEE",
                "RIGHT_KNEE",
                "LEFT_ANKLE",
                "RIGHT_ANKLE",
            ]
        else:
            joints_to_analyze = custom_joints

        results = run_joint_level_analysis(
            dataset_name=self.dataset_name,
            joint_names=joints_to_analyze,
            output_dir=output_dir,
            save_results=self.settings.get("save_results", True),
            sampling_radius=sampling_radius,
        )

        return bool(results)

    def run_complete_analysis_suite(self, include_multi: bool = None) -> dict:
        """Run complete analysis suite with remaining pipelines.

        Args:
            include_multi: Deprecated parameter (kept for compatibility)

        Returns:
            dict: Results summary from all analyses
        """
        print("=== COMPLETE ANALYSIS SUITE ===")

        results = {
            "standard_analysis": False,
            "joint_level_analysis": False,
        }

        # Run standard analysis
        try:
            results["standard_analysis"] = self.run_standard_analysis()
        except Exception as e:
            print(f"Standard analysis failed: {e}")

        # Run joint-level analysis
        try:
            results["joint_level_analysis"] = self.run_joint_level_analysis()
        except Exception as e:
            print(f"Joint-level analysis failed: {e}")

        # Print summary
        self._print_analysis_summary(results)

        return results

    def _print_analysis_summary(self, results: dict):
        """Print summary of analysis results.

        Args:
            results: Dictionary containing analysis results
        """
        print("\n" + "=" * 70)
        print("ANALYSIS SUITE SUMMARY")
        print("=" * 70)

        for analysis_type, success in results.items():
            status = "✓ SUCCESS" if success else "✗ FAILED"
            print(f"{analysis_type.replace('_', ' ').title():<30} {status}")

        total_success = sum(results.values())
        total_run = len([v for v in results.values() if v is not False])

        print("-" * 70)
        print(f"Overall Success Rate: {total_success}/{total_run}")
        print("=" * 70)

    def get_available_analyses(self) -> list:
        """Get list of available analysis types.

        Returns:
            list: List of available analysis types
        """
        analyses = ["standard_analysis", "joint_level_analysis"]

        return analyses

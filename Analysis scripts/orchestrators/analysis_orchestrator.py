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
from utils.config_extractor import extract_joint_analysis_config, get_analysis_settings
from runners.single_analysis_runner import run_single_analysis
from runners.multi_analysis_runner import run_multi_analysis
from joint_analysis_runner import run_joint_analysis
from per_video_joint_analysis_runner import run_per_video_analysis


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

    def run_joint_analysis(
        self,
        custom_joints: Optional[list] = None,
        custom_thresholds: Optional[list] = None,
        output_dir: Optional[str] = None,
    ) -> bool:
        """Run joint analysis pipeline.

        Args:
            custom_joints: Custom list of joints (overrides config)
            custom_thresholds: Custom list of thresholds (overrides config)
            output_dir: Custom output directory

        Returns:
            bool: True if analysis completed successfully
        """
        print("=== JOINT ANALYSIS PIPELINE ===")

        # Extract from config if not provided
        if custom_joints is None or custom_thresholds is None:
            config_joints, config_thresholds = extract_joint_analysis_config(
                self.dataset_name
            )
            joints_to_analyze = custom_joints or config_joints
            pck_thresholds = custom_thresholds or config_thresholds
        else:
            joints_to_analyze = custom_joints
            pck_thresholds = custom_thresholds

        return run_joint_analysis(
            dataset_name=self.dataset_name,
            joints_to_analyze=joints_to_analyze,
            pck_thresholds=pck_thresholds,
            output_dir=output_dir,
            save_results=self.settings.get("save_results", True),
        )

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

    def run_multi_analysis(self, metrics_config: Optional[dict] = None) -> bool:
        """Run multi-analysis pipeline.

        Args:
            metrics_config: Configuration for metrics

        Returns:
            bool: True if analysis completed successfully
        """
        print("=== MULTI-ANALYSIS PIPELINE ===")

        if metrics_config is None:
            metrics_config = {"brightness": "get_brightness_data"}

        return run_multi_analysis(
            dataset_name=self.dataset_name,
            metrics_config=metrics_config,
            analysis_config=self.analysis_config,
        )

    def run_per_video_analysis(
        self,
        custom_joints: Optional[list] = None,
        output_dir: Optional[str] = None,
        sampling_radius: int = 3,
    ) -> bool:
        """Run per-video joint brightness analysis pipeline.

        Args:
            custom_joints: Custom list of joints (uses default if None)
            output_dir: Custom output directory
            sampling_radius: Radius for brightness sampling around joints

        Returns:
            bool: True if analysis completed successfully
        """
        print("=== PER-VIDEO JOINT BRIGHTNESS ANALYSIS ===")

        # Extract from config if not provided
        if custom_joints is None:
            config_joints, _ = extract_joint_analysis_config(self.dataset_name)
            joints_to_analyze = config_joints
        else:
            joints_to_analyze = custom_joints

        results = run_per_video_analysis(
            dataset_name=self.dataset_name,
            joint_names=joints_to_analyze,
            output_dir=output_dir,
            save_results=self.settings.get("save_results", True),
            sampling_radius=sampling_radius,
        )

        return bool(results)

    def run_complete_analysis_suite(self, include_multi: bool = None) -> dict:
        """Run complete analysis suite with all pipelines.

        Args:
            include_multi: Whether to include multi-analysis (auto-detect if None)

        Returns:
            dict: Results summary from all analyses
        """
        print("=== COMPLETE ANALYSIS SUITE ===")

        results = {
            "joint_analysis": False,
            "standard_analysis": False,
            "multi_analysis": False,
            "per_video_analysis": False,
        }

        # Run joint analysis
        try:
            results["joint_analysis"] = self.run_joint_analysis()
        except Exception as e:
            print(f"Joint analysis failed: {e}")

        # Run standard analysis
        try:
            results["standard_analysis"] = self.run_standard_analysis()
        except Exception as e:
            print(f"Standard analysis failed: {e}")

        # Run per-video analysis
        try:
            results["per_video_analysis"] = self.run_per_video_analysis()
        except Exception as e:
            print(f"Per-video analysis failed: {e}")

        # Determine if multi-analysis should be run
        if include_multi is None:
            include_multi = (
                self.analysis_config
                and self.analysis_config.is_multi_analysis_enabled()
            )

        # Run multi-analysis if enabled
        if include_multi:
            try:
                results["multi_analysis"] = self.run_multi_analysis()
            except Exception as e:
                print(f"Multi-analysis failed: {e}")

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
        analyses = ["joint_analysis", "standard_analysis", "per_video_analysis"]

        if self.analysis_config and self.analysis_config.is_multi_analysis_enabled():
            analyses.append("multi_analysis")

        return analyses

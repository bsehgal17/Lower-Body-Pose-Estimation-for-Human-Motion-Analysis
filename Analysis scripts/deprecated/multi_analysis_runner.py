"""
Multi Analysis Runner

Handles multi-analysis pipeline execution with various analysis scenarios.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config import ConfigManager
from core.pipeline_manager import AnalysisPipeline
from core.multi_analysis_pipeline import MultiAnalysisPipeline


def run_multi_analysis(
    dataset_name: str, metrics_config: dict, analysis_config
) -> bool:
    """Run multi-analysis pipeline.

    Args:
        dataset_name: Name of the dataset to analyze
        metrics_config: Configuration for metrics
        analysis_config: Analysis configuration object

    Returns:
        bool: True if analysis completed successfully
    """
    print("Running Multi-Analysis Pipeline")
    print("=" * 70)

    try:
        # Create base pipeline for shared components
        base_pipeline = AnalysisPipeline(dataset_name)
        dataset_config = ConfigManager.load_config(dataset_name)

        # Create multi-analysis pipeline
        multi_pipeline = MultiAnalysisPipeline(
            base_pipeline.config, base_pipeline.data_processor, base_pipeline.timestamp
        )

        # Run additional multi-analysis scenarios
        multi_pipeline.run_multi_analysis(
            analysis_config, dataset_config, metrics_config
        )

        print("Multi-analysis completed successfully")
        return True

    except Exception as e:
        print(f"ERROR: Multi-analysis failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def run_custom_multi_analysis(
    dataset_name: str,
    metrics_config: dict = None,
    analysis_scenarios: list = None,
    comparison_groups: list = None,
) -> bool:
    """Run multi-analysis with custom scenarios.

    Args:
        dataset_name: Name of the dataset to analyze
        metrics_config: Configuration for metrics
        analysis_scenarios: List of analysis scenarios to run
        comparison_groups: List of comparison groups for analysis

    Returns:
        bool: True if analysis completed successfully
    """
    if metrics_config is None:
        metrics_config = {"brightness": "get_brightness_data"}

    try:
        base_pipeline = AnalysisPipeline(dataset_name)
        dataset_config = ConfigManager.load_config(dataset_name)

        multi_pipeline = MultiAnalysisPipeline(
            base_pipeline.config, base_pipeline.data_processor, base_pipeline.timestamp
        )

        # Custom multi-analysis logic could be added here
        # For now, use the standard multi-analysis
        multi_pipeline.run_multi_analysis(None, dataset_config, metrics_config)

        return True

    except Exception as e:
        print(f"ERROR: Custom multi-analysis failed: {e}")
        return False


def check_multi_analysis_requirements(dataset_name: str) -> bool:
    """Check if requirements for multi-analysis are met.

    Args:
        dataset_name: Name of the dataset to check

    Returns:
        bool: True if requirements are met
    """
    try:
        # Check if dataset config can be loaded
        ConfigManager.load_config(dataset_name)

        # Add specific requirement checks here
        # For example: check if required data files exist, etc.

        return True

    except Exception as e:
        print(f"WARNING: Multi-analysis requirements check failed: {e}")
        return False

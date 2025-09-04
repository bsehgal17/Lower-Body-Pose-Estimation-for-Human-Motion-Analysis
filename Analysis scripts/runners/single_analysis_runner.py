"""
Single Analysis Runner

Handles standard single analysis pipeline execution.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from core.pipeline_manager import AnalysisPipeline


def run_single_analysis(
    dataset_name: str, metrics_config: dict, analysis_config
) -> bool:
    """Run single analysis pipeline.

    Args:
        dataset_name: Name of the dataset to analyze
        metrics_config: Configuration for metrics
        analysis_config: Analysis configuration object

    Returns:
        bool: True if analysis completed successfully
    """
    print("Running Single Analysis Pipeline")
    print("=" * 70)

    try:
        pipeline = AnalysisPipeline(dataset_name)

        # Import here to avoid circular imports
        from analyzers.analyzer_factory import AnalyzerFactory

        per_frame_analysis_types = AnalyzerFactory.get_available_analyzers()

        pipeline.run_complete_analysis(
            metrics_config=metrics_config,
            run_overall=True,
            run_per_video=True,
            run_per_frame=True,
            per_frame_analysis_types=per_frame_analysis_types,
        )

        print("Single analysis completed successfully")
        return True

    except Exception as e:
        print(f"ERROR: Single analysis failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def run_single_analysis_with_options(
    dataset_name: str,
    metrics_config: dict = None,
    run_overall: bool = True,
    run_per_video: bool = True,
    run_per_frame: bool = True,
    analysis_types: list = None,
) -> bool:
    """Run single analysis with custom options.

    Args:
        dataset_name: Name of the dataset to analyze
        metrics_config: Configuration for metrics (default: brightness)
        run_overall: Whether to run overall analysis
        run_per_video: Whether to run per-video analysis
        run_per_frame: Whether to run per-frame analysis
        analysis_types: List of analysis types to run

    Returns:
        bool: True if analysis completed successfully
    """
    if metrics_config is None:
        metrics_config = {"brightness": "get_brightness_data"}

    try:
        pipeline = AnalysisPipeline(dataset_name)

        if analysis_types is None:
            from analyzers.analyzer_factory import AnalyzerFactory

            analysis_types = AnalyzerFactory.get_available_analyzers()

        pipeline.run_complete_analysis(
            metrics_config=metrics_config,
            run_overall=run_overall,
            run_per_video=run_per_video,
            run_per_frame=run_per_frame,
            per_frame_analysis_types=analysis_types,
        )

        return True

    except Exception as e:
        print(f"ERROR: Custom single analysis failed: {e}")
        return False

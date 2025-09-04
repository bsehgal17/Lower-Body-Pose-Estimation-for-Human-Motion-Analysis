"""
Simplified Analysis Main Entry Point

Uses modular components for clean and maintainable analysis pipeline.
"""

from config import ConfigManager, load_dataset_analysis_config

# from analyzers.analyzer_factory import AnalyzerFactory
from core.pipeline_manager import AnalysisPipeline
from core.multi_analysis_pipeline import MultiAnalysisPipeline


def run_single_analysis(dataset_name: str, metrics_config: dict, analysis_config):
    """Run single analysis pipeline."""
    print("Running Single Analysis Pipeline")
    print("=" * 70)

    pipeline = AnalysisPipeline(dataset_name)
    from analyzers.analyzer_factory import AnalyzerFactory

    per_frame_analysis_types = AnalyzerFactory.get_available_analyzers()

    pipeline.run_complete_analysis(
        metrics_config=metrics_config,
        run_overall=True,
        run_per_video=True,
        run_per_frame=True,
        per_frame_analysis_types=per_frame_analysis_types,
    )

    print("Single analysis completed")


def run_multi_analysis(dataset_name: str, metrics_config: dict, analysis_config):
    """Run multi-analysis pipeline."""
    # Create base pipeline for shared components
    base_pipeline = AnalysisPipeline(dataset_name)
    dataset_config = ConfigManager.load_config(dataset_name)

    # Create multi-analysis pipeline
    multi_pipeline = MultiAnalysisPipeline(
        base_pipeline.config, base_pipeline.data_processor, base_pipeline.timestamp
    )

    # Run additional multi-analysis scenarios
    multi_pipeline.run_multi_analysis(
        analysis_config, dataset_config, metrics_config)


def main():
    """Main entry point for analysis."""
    # Configuration
    dataset_name = "movi"
    metrics_config = {
        "brightness": "get_brightness_data",
    }

    # Load analysis configuration from dataset-specific config only
    analysis_config = load_dataset_analysis_config(dataset_name)

    try:
        # Check if multi-analysis is enabled in config
        if analysis_config.is_multi_analysis_enabled():
            run_multi_analysis(dataset_name, metrics_config, analysis_config)
        else:
            run_single_analysis(dataset_name, metrics_config, analysis_config)

    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

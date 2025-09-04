"""
Configuration management components.
"""

from .dataset_config import DatasetConfig
from .config_factory import ConfigFactory
from .config_manager import ConfigManager
from .analysis_config import (
    AnalysisConfig,
    AnalysisConfigLoader,
    load_analysis_config,
    load_dataset_analysis_config,
)


class ConfigurationTester:
    """Tests configuration loading and components."""

    def test_dataset_configuration(self, dataset_name: str):
        """Test configuration for a specific dataset."""
        print(f"\nTesting {dataset_name.upper()} configuration:")
        print("-" * 40)

        try:
            config = ConfigManager.load_config(dataset_name)

            print("Configuration loaded successfully!")
            print(f"   Name: {config.name}")
            print(f"   Model: {config.model}")
            print(f"   PCK Overall columns: {config.pck_overall_score_columns}")
            print(f"   PCK Per-frame columns: {config.pck_per_frame_score_columns}")

            # Check for dataset-specific analysis config
            if hasattr(config, "analysis_config") and config.analysis_config:
                pck_brightness_config = config.analysis_config.get("pck_brightness", {})
                if pck_brightness_config:
                    score_groups = pck_brightness_config.get("score_groups", {})
                    default_group = pck_brightness_config.get(
                        "default_score_group", "all"
                    )
                    print(f"   Available score groups: {list(score_groups.keys())}")
                    print(f"   Default score group: {default_group}")

            validation_result = config.validate()
            print(
                f"   Configuration validation: {'Passed' if validation_result else 'Failed'}"
            )

        except Exception as e:
            print(f"Error loading {dataset_name} configuration: {e}")

    def test_analysis_components(self):
        """Test analysis components availability."""
        from analyzers import AnalyzerFactory
        from visualizers import VisualizationFactory

        analysis_config = load_analysis_config()
        per_frame_analysis_types = AnalyzerFactory.get_available_analyzers()

        print(f"Available analyzers: {AnalyzerFactory.get_available_analyzers()}")
        print(
            f"Available visualizers: {VisualizationFactory.get_available_visualizers()}"
        )
        print(f"Will run per-frame analysis types: {per_frame_analysis_types}")
        print(
            f"Global analysis score groups: {analysis_config.get_available_score_groups()}"
        )


__all__ = [
    "DatasetConfig",
    "ConfigFactory",
    "ConfigManager",
    "AnalysisConfig",
    "AnalysisConfigLoader",
    "load_analysis_config",
    "load_dataset_analysis_config",
    "ConfigurationTester",
]

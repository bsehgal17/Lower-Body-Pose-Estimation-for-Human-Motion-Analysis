"""
Configuration Manager Module

Handles loading and management of analysis configurations.
"""

from config import ConfigManager, load_analysis_config
from analyzers import AnalyzerFactory
from visualizers import VisualizationFactory


class AnalysisConfigManager:
    """Manages analysis configurations and settings."""

    @staticmethod
    def load_dataset_config(dataset_name: str):
        """Load configuration for a specific dataset."""
        return ConfigManager.load_config(dataset_name)

    @staticmethod
    def load_analysis_config():
        """Load general analysis configuration."""
        return load_analysis_config()

    @staticmethod
    def get_metrics_config():
        """Get default metrics configuration."""
        return {
            "brightness": "get_brightness_data",
        }

    @staticmethod
    def test_configurations():
        """Test YAML configuration loading."""
        analysis_config = load_analysis_config()

        print("🧪 Testing YAML-Based Dataset Configuration")
        print("=" * 60)

        datasets_to_test = ["humaneva", "movi"]

        for test_dataset in datasets_to_test:
            print(f"\n📋 Testing {test_dataset.upper()} configuration:")
            print("-" * 40)

            try:
                config = ConfigManager.load_config(test_dataset)

                print("✅ Configuration loaded successfully!")
                print(f"   Name: {config.name}")
                print(f"   Model: {config.model}")
                print(f"   PCK Overall columns: {config.pck_overall_score_columns}")
                print(f"   PCK Per-frame columns: {config.pck_per_frame_score_columns}")

                # Check for dataset-specific analysis config
                if hasattr(config, "analysis_config") and config.analysis_config:
                    pck_brightness_config = config.analysis_config.get(
                        "pck_brightness", {}
                    )
                    if pck_brightness_config:
                        score_groups = pck_brightness_config.get("score_groups", {})
                        default_group = pck_brightness_config.get(
                            "default_score_group", "all"
                        )
                        print(f"   Available score groups: {list(score_groups.keys())}")
                        print(f"   Default score group: {default_group}")

                validation_result = config.validate()
                print(
                    f"   Configuration validation: {'✅ Passed' if validation_result else '❌ Failed'}"
                )

            except Exception as e:
                print(f"❌ Error loading {test_dataset} configuration: {e}")

        return analysis_config


class ConfigurationTester:
    """Tests configuration loading and components."""

    def test_dataset_configuration(self, dataset_name: str):
        """Test configuration for a specific dataset."""
        print(f"\n📋 Testing {dataset_name.upper()} configuration:")
        print("-" * 40)

        try:
            config = ConfigManager.load_config(dataset_name)

            print("✅ Configuration loaded successfully!")
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
                f"   Configuration validation: {'✅ Passed' if validation_result else '❌ Failed'}"
            )

        except Exception as e:
            print(f"❌ Error loading {dataset_name} configuration: {e}")

    def test_analysis_components(self):
        """Test analysis components availability."""
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

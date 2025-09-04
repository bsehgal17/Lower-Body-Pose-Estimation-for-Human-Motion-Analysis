"""
Analysis Configuration Loader.

Loads and manages analysis-specific configurations from YAML files.
"""

import yaml
import os
from typing import Dict, Any, List, Optional


class AnalysisConfig:
    """Configuration class for analysis settings."""

    def __init__(self, config_dict: Dict[str, Any]):
        """Initialize with configuration dictionary."""
        self.config_dict = config_dict
        self._load_config()

    def _load_config(self):
        """Load configuration values."""
        # PCK Brightness Analysis Config
        pck_config = self.config_dict.get("analysis", {}).get("pck_brightness", {})

        self.pck_score_groups = pck_config.get("score_groups", {})
        self.default_score_group = pck_config.get("default_score_group", "all")
        self.multi_analysis = pck_config.get("multi_analysis", {})
        self.visualization = pck_config.get("visualization", {})
        self.export = pck_config.get("export", {})

        # General settings
        general_config = self.config_dict.get("general", {})
        self.output_prefix = general_config.get("output_prefix", "analysis")
        self.timestamp_format = general_config.get("timestamp_format", "%Y%m%d_%H%M%S")
        self.create_summary_report = general_config.get("create_summary_report", True)
        self.verbose_logging = general_config.get("verbose_logging", True)

        # Other analysis types
        self.statistical = self.config_dict.get("analysis", {}).get("statistical", {})
        self.distribution = self.config_dict.get("analysis", {}).get("distribution", {})
        self.correlation = self.config_dict.get("analysis", {}).get("correlation", {})

    def get_score_group(self, group_name: str) -> Optional[List[int]]:
        """Get score group by name."""
        return self.pck_score_groups.get(group_name)

    def get_available_score_groups(self) -> List[str]:
        """Get list of available score group names."""
        return list(self.pck_score_groups.keys())

    def get_multi_analysis_scenarios(self) -> List[Dict[str, Any]]:
        """Get multi-analysis scenarios if enabled."""
        if self.multi_analysis.get("enabled", False):
            return self.multi_analysis.get("scenarios", [])
        return []

    def is_multi_analysis_enabled(self) -> bool:
        """Check if multi-analysis is enabled."""
        return self.multi_analysis.get("enabled", False)

    def should_create_individual_plots(self) -> bool:
        """Check if individual plots should be created."""
        return self.visualization.get("create_individual_plots", True)

    def should_create_combined_plots(self) -> bool:
        """Check if combined plots should be created."""
        return self.visualization.get("create_combined_plots", True)

    def should_create_comparison_plots(self) -> bool:
        """Check if comparison plots should be created."""
        return self.visualization.get("create_comparison_plots", True)

    def get_save_format(self) -> str:
        """Get visualization save format."""
        return self.visualization.get("save_format", "svg")

    def is_csv_export_enabled(self) -> bool:
        """Check if CSV export is enabled."""
        return self.export.get("csv_enabled", True)

    def is_json_export_enabled(self) -> bool:
        """Check if JSON export is enabled."""
        return self.export.get("json_enabled", True)

    def should_create_separate_files_per_group(self) -> bool:
        """Check if separate files should be created per score group."""
        return self.export.get("separate_files_per_group", True)


class AnalysisConfigLoader:
    """Loader for analysis configurations."""

    @staticmethod
    def load_config(config_path: str = None) -> AnalysisConfig:
        """
        Load analysis configuration from YAML file.

        Args:
            config_path: Path to configuration file. If None, uses default location.

        Returns:
            AnalysisConfig instance
        """
        if config_path is None:
            # Default path relative to this file - look in Analysis scripts/config_yamls
            current_dir = os.path.dirname(os.path.abspath(__file__))
            analysis_scripts_dir = os.path.dirname(current_dir)
            config_path = os.path.join(
                analysis_scripts_dir, "config_yamls", "analysis_config.yaml"
            )

        try:
            with open(config_path, "r") as file:
                config_dict = yaml.safe_load(file)

            return AnalysisConfig(config_dict)

        except FileNotFoundError:
            print(f"Warning: Analysis config file not found at {config_path}")
            print("Using default configuration.")
            return AnalysisConfig({})

        except yaml.YAMLError as e:
            print(f"Error parsing YAML config: {e}")
            print("Using default configuration.")
            return AnalysisConfig({})

    @staticmethod
    def create_default_config() -> AnalysisConfig:
        """Create a default analysis configuration."""
        default_config = {
            "analysis": {
                "pck_brightness": {
                    "score_groups": {
                        "all": None,
                        "extremes": [0, 100],
                        "high_performance": [80, 85, 90, 95, 100],
                    },
                    "default_score_group": "all",
                    "multi_analysis": {"enabled": False, "scenarios": []},
                    "visualization": {
                        "create_individual_plots": True,
                        "create_combined_plots": True,
                        "save_format": "svg",
                    },
                    "export": {
                        "csv_enabled": True,
                        "json_enabled": True,
                        "separate_files_per_group": True,
                    },
                }
            },
            "general": {
                "output_prefix": "analysis",
                "timestamp_format": "%Y%m%d_%H%M%S",
                "create_summary_report": True,
                "verbose_logging": True,
            },
        }

        return AnalysisConfig(default_config)


# Convenience function for easy import
def load_analysis_config(
    config_path: str = None, dataset_name: str = None
) -> AnalysisConfig:
    """
    Load analysis configuration from dataset-specific config files only.

    Args:
        config_path: Optional path to specific config file
        dataset_name: Dataset name to load dataset-specific config

    Returns:
        AnalysisConfig instance
    """
    if config_path:
        return AnalysisConfigLoader.load_config(config_path)

    if dataset_name:
        return load_dataset_analysis_config(dataset_name)

    # If no dataset specified, return default config
    print("[WARNING] No dataset name specified, using default configuration")
    return AnalysisConfigLoader.create_default_config()


def load_dataset_analysis_config(dataset_name: str) -> AnalysisConfig:
    """
    Load analysis configuration from dataset-specific config file only.

    Args:
        dataset_name: Name of the dataset

    Returns:
        AnalysisConfig instance with dataset-specific settings
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    analysis_scripts_dir = os.path.dirname(current_dir)

    # Load dataset-specific config
    dataset_config_path = os.path.join(
        analysis_scripts_dir, "config_yamls", f"{dataset_name}_config.yaml"
    )

    if not os.path.exists(dataset_config_path):
        raise FileNotFoundError(
            f"Dataset configuration file not found: {dataset_config_path}\n"
            f"Please create {dataset_name}_config.yaml with analysis configuration."
        )

    try:
        with open(dataset_config_path, "r") as file:
            dataset_config_dict = yaml.safe_load(file)

        # Check if dataset config has analysis section
        if "analysis" not in dataset_config_dict:
            raise ValueError(
                f"Dataset config {dataset_name}_config.yaml is missing 'analysis' section.\n"
                f"Please add analysis configuration to the dataset config file."
            )

        print(
            f"[SUCCESS] Using dataset-specific analysis config: {dataset_name}_config.yaml"
        )
        return AnalysisConfig(dataset_config_dict)

    except Exception as e:
        raise Exception(
            f"Error loading dataset config {dataset_config_path}: {e}\n"
            f"Please ensure {dataset_name}_config.yaml exists and has valid analysis configuration."
        )


def load_analysis_config_with_dataset_priority(dataset_name: str) -> AnalysisConfig:
    """
    Load analysis configuration from dataset-specific config file only.

    This function is now equivalent to load_dataset_analysis_config.

    Args:
        dataset_name: Name of the dataset

    Returns:
        AnalysisConfig instance with dataset-specific settings
    """
    return load_dataset_analysis_config(dataset_name)

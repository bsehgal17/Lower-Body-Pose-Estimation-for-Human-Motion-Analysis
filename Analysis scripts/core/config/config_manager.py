"""
Configuration manager for loading and validating configurations.
"""

from .dataset_config import DatasetConfig
from .config_factory import ConfigFactory


class ConfigManager:
    """Enhanced configuration manager."""

    @staticmethod
    def load_config(dataset_name: str) -> DatasetConfig:
        """Load configuration for a dataset."""
        try:
            config = ConfigFactory.create_config(dataset_name)
            print(f"Configuration loaded for {dataset_name.upper()} dataset")
            return config
        except ValueError as e:
            print(f"Error loading configuration: {e}")
            raise

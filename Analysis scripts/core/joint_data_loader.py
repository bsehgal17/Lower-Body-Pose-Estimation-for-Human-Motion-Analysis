"""
Joint Analysis Data Loader

Handles data loading and validation for joint-wise PCK analysis.
"""

import pandas as pd
from typing import Optional

from core.data_processor import DataProcessor
from config.config_manager import ConfigManager


class JointDataLoader:
    """Handles loading and validation of joint analysis data."""

    def __init__(self, dataset_name: str):
        """Initialize the data loader.

        Args:
            dataset_name: Name of the dataset to load
        """
        self.dataset_name = dataset_name
        self.config = None
        self.data_processor = None

    def setup_configuration(self) -> bool:
        """Setup configuration for the dataset.

        Returns:
            bool: True if configuration loaded successfully
        """
        try:
            print(f"Setting up configuration for dataset: {self.dataset_name}")

            self.config = ConfigManager.load_config(self.dataset_name)

            if self.config is None:
                print(
                    f"ERROR: Could not load configuration for dataset '{self.dataset_name}'"
                )
                print("Available datasets: movi, humaneva")
                return False

            print(f"Configuration loaded successfully for {self.dataset_name}")
            return True

        except Exception as e:
            print(f"ERROR: Failed to setup configuration: {e}")
            return False

    def load_pck_data(self) -> Optional[pd.DataFrame]:
        """Load PCK data for joint analysis.

        Returns:
            pd.DataFrame: Loaded PCK data or None if failed
        """
        try:
            print("Loading PCK data for joint analysis...")

            # Initialize data processor
            self.data_processor = DataProcessor(self.config)

            # Load jointwise PCK scores
            pck_data = self.data_processor.load_pck_jointwise_scores()

            if pck_data is None or pck_data.empty:
                print("ERROR: No PCK data loaded")
                return None

            print(f"PCK data loaded successfully. Shape: {pck_data.shape}")
            return pck_data

        except Exception as e:
            print(f"ERROR: Failed to load PCK data: {e}")
            import traceback

            traceback.print_exc()
            return None

    def validate_data(
        self, pck_data: pd.DataFrame, joints_to_analyze: list, pck_thresholds: list
    ) -> bool:
        """Validate that the data contains required columns.

        Args:
            pck_data: The PCK data to validate
            joints_to_analyze: List of joints that should be present
            pck_thresholds: List of PCK thresholds that should be present

        Returns:
            bool: True if data is valid
        """
        try:
            print("Validating PCK data...")

            required_columns = []
            for joint in joints_to_analyze:
                for threshold in pck_thresholds:
                    required_columns.append(f"{joint}_pck_{threshold:g}")

            missing_columns = [
                col for col in required_columns if col not in pck_data.columns
            ]

            if missing_columns:
                print(f"ERROR: Missing required columns: {missing_columns}")
                print(f"Available columns: {list(pck_data.columns)}")
                return False

            print("Data validation successful")
            return True

        except Exception as e:
            print(f"ERROR: Failed to validate data: {e}")
            return False

    def load_and_validate(
        self, joints_to_analyze: list, pck_thresholds: list
    ) -> Optional[pd.DataFrame]:
        """Complete data loading and validation pipeline.

        Args:
            joints_to_analyze: List of joints to analyze
            pck_thresholds: List of PCK thresholds

        Returns:
            pd.DataFrame: Validated PCK data or None if failed
        """
        # Setup configuration
        if not self.setup_configuration():
            return None

        # Load data
        pck_data = self.load_pck_data()
        if pck_data is None:
            return None

        # Validate data
        if not self.validate_data(pck_data, joints_to_analyze, pck_thresholds):
            return None

        return pck_data

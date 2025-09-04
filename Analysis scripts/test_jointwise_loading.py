#!/usr/bin/env python3
"""
Test script to verify jointwise PCK data loading functionality.
"""

import sys
import os

# Add Analysis scripts to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.config_manager import ConfigManager
from core.data_processor import DataProcessor


def test_jointwise_loading(dataset_name="movi"):
    """Test loading jointwise PCK data."""
    print(f"Testing jointwise PCK data loading for {dataset_name}")
    print("=" * 60)

    try:
        # Load configuration
        print("1. Loading configuration...")
        config = ConfigManager.load_config(dataset_name)

        if config is None:
            print(f"ERROR: Could not load configuration for {dataset_name}")
            return False

        print("   Configuration loaded successfully")
        print(f"   Jointwise columns: {config.pck_jointwise_score_columns}")

        # Create data processor
        print("\n2. Creating data processor...")
        data_processor = DataProcessor(config)

        # Test loading jointwise scores
        print("\n3. Loading jointwise PCK scores...")
        try:
            jointwise_df = data_processor.load_pck_jointwise_scores()

            if jointwise_df is not None:
                print(f"   SUCCESS: Loaded {len(jointwise_df)} records")
                print(f"   Columns: {list(jointwise_df.columns)}")
                print(f"   Shape: {jointwise_df.shape}")

                # Show sample data
                print("\nSample data (first 3 rows):")
                print(jointwise_df.head(3).to_string())

                return True
            else:
                print("   ERROR: No data loaded")
                return False

        except Exception as e:
            print(f"   ERROR loading jointwise data: {e}")
            return False

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Test both datasets
    for dataset in ["movi", "humaneva"]:
        success = test_jointwise_loading(dataset)
        print(f"\n{dataset.upper()} test: {'PASSED' if success else 'FAILED'}")
        print("-" * 60)

"""
Simple PCK Data Loader Script

Loads PCK scores from Excel files with minimal configuration.
Focus: Just data loading, nothing else.
"""

import sys
import os
import pandas as pd
from typing import Optional

# Add the Analysis scripts directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import ConfigManager
from processors import PCKDataLoader


class SimplePCKDataLoader:
    """Simple wrapper for loading PCK data."""

    def __init__(self, dataset_name: str):
        """Initialize with dataset name."""
        self.dataset_name = dataset_name
        self.config = ConfigManager.load_config(dataset_name)
        self.loader = PCKDataLoader(self.config)

    def load_overall_data(self) -> Optional[pd.DataFrame]:
        """Load overall PCK scores."""
        print(f"Loading overall PCK data for {self.dataset_name}...")
        data = self.loader.load_overall_scores()

        if data is not None:
            print(f"✅ Loaded {len(data)} overall records")
            print(f"   Columns: {list(data.columns)}")
        else:
            print("❌ Failed to load overall data")

        return data

    def load_per_frame_data(self) -> Optional[pd.DataFrame]:
        """Load per-frame PCK scores."""
        print(f"Loading per-frame PCK data for {self.dataset_name}...")
        data = self.loader.load_per_frame_scores()

        if data is not None:
            print(f"✅ Loaded {len(data)} per-frame records")
            print(f"   Columns: {list(data.columns)}")
            print(f"   PCK score columns: {self.config.pck_per_frame_score_columns}")
        else:
            print("❌ Failed to load per-frame data")

        return data

    def preview_data(self, data_type: str = "per_frame", num_rows: int = 5):
        """Preview the data."""
        if data_type == "per_frame":
            data = self.load_per_frame_data()
        else:
            data = self.load_overall_data()

        if data is not None:
            print(f"\nData Preview ({data_type}):")
            print("-" * 50)
            print(data.head(num_rows))
            print(f"\nData Shape: {data.shape}")


def main():
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Simple PCK Data Loader")
    parser.add_argument("dataset", help="Dataset name (e.g., 'movi', 'humaneva')")
    parser.add_argument(
        "--type",
        choices=["overall", "per_frame", "both"],
        default="both",
        help="Type of data to load",
    )
    parser.add_argument("--preview", action="store_true", help="Show data preview")

    args = parser.parse_args()

    try:
        loader = SimplePCKDataLoader(args.dataset)

        if args.type in ["overall", "both"]:
            overall_data = loader.load_overall_data()
            if args.preview and overall_data is not None:
                loader.preview_data("overall")

        if args.type in ["per_frame", "both"]:
            per_frame_data = loader.load_per_frame_data()
            if args.preview and per_frame_data is not None:
                loader.preview_data("per_frame")

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

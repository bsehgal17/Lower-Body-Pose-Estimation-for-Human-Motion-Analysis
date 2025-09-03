"""
PCK frame count analyzer.
"""

from base_classes import BaseAnalyzer
import pandas as pd
import os


class PCKFrameCountAnalyzer(BaseAnalyzer):
    """Analyzer for counting frames at each PCK score level."""

    def __init__(self, config):
        """Initialize with configuration."""
        self.config = config

    def analyze(self, data: pd.DataFrame, metric_name: str) -> pd.DataFrame:
        """
        Count frames at each PCK score level.

        Args:
            data: DataFrame with PCK score columns
            metric_name: Not used for this analyzer

        Returns:
            DataFrame with PCK score frame counts
        """
        print("\n" + "=" * 50)
        print("Running PCK Score Frame Count...")

        if not hasattr(self.config, "pck_per_frame_score_columns"):
            raise ValueError("Config must have pck_per_frame_score_columns attribute.")

        results = []

        for pck_col in self.config.pck_per_frame_score_columns:
            if pck_col not in data.columns:
                print(f"⚠️ Skipping {pck_col} (not in DataFrame)")
                continue

            # Count how many frames at each PCK value
            counts = data[pck_col].value_counts().sort_index().reset_index()
            counts.columns = ["PCK_Score", "Frame_Count"]
            counts["PCK_Column"] = pck_col

            results.append(counts)

        if not results:
            print("No valid PCK columns found. Exiting...")
            return pd.DataFrame()

        # Combine results from all PCK columns
        final_df = pd.concat(results, ignore_index=True)

        # Save results
        self._save_results(final_df)

        print("=" * 50 + "\nPCK Score Frame Count Completed.")
        return final_df

    def _save_results(self, results_df: pd.DataFrame):
        """Save PCK frame count results to Excel."""
        summary_file = os.path.join(
            self.config.save_folder, "pck_score_frame_counts.xlsx"
        )

        os.makedirs(self.config.save_folder, exist_ok=True)

        if os.path.exists(summary_file):
            existing_df = pd.read_excel(summary_file)
            combined_df = pd.concat([existing_df, results_df], ignore_index=True)
            combined_df.to_excel(summary_file, index=False)
            print(f"Updated PCK score frame counts saved to '{summary_file}'")
        else:
            results_df.to_excel(summary_file, index=False)
            print(f"PCK score frame counts created at '{summary_file}'")

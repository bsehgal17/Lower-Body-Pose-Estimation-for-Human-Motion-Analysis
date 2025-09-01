"""
Modular analyzers replacing original analysis scripts.
"""

from base_classes import BaseAnalyzer
import pandas as pd
from typing import Dict, Any, List
import scipy.stats as stats
import os


class ANOVAAnalyzer(BaseAnalyzer):
    """Analyzer for performing ANOVA tests."""

    def analyze(self, data: pd.DataFrame, metric_name: str) -> Dict[str, Any]:
        """Perform ANOVA analysis on the data."""
        print(f"\nRunning ANOVA Test for Metric: {metric_name.title()}")

        cleaned_data = self._clean_data(data, metric_name)
        binned_data = self._create_bins(cleaned_data, metric_name)
        results = self._run_anova_tests(binned_data, metric_name)
        self._save_results(results, metric_name)

        return results

    def _clean_data(self, df: pd.DataFrame, metric_name: str) -> pd.DataFrame:
        """Clean the data for analysis."""
        cleaned_df = df.copy()

        if cleaned_df.isnull().values.any():
            print("Warning: Missing values found. Filling with 0...")
            columns_to_fill = [metric_name] + self.config.pck_per_frame_score_columns
            cleaned_df.loc[:, columns_to_fill] = cleaned_df.loc[
                :, columns_to_fill
            ].fillna(0)
            cleaned_df.dropna(inplace=True)

        return cleaned_df

    def _create_bins(self, df: pd.DataFrame, metric_name: str) -> pd.DataFrame:
        """Create bins for the metric."""
        if metric_name == "brightness":
            bins = [0, 50, 100, 150, 200, 255]
            labels = ["Low", "Medium-Low", "Medium", "Medium-High", "High"]
        else:
            bins = pd.qcut(df[metric_name], q=5, retbins=True, labels=False)[1]
            labels = [f"Bin {i + 1}" for i in range(5)]

        df[f"{metric_name}_bin"] = pd.cut(
            df[metric_name], bins=bins, labels=labels, right=False
        )
        return df

    def _run_anova_tests(
        self, df: pd.DataFrame, metric_name: str
    ) -> List[Dict[str, Any]]:
        """Run ANOVA tests for each PCK column."""
        results = []
        bin_column = f"{metric_name}_bin"
        unique_bins = df[bin_column].dropna().unique()

        for pck_col in self.config.pck_per_frame_score_columns:
            groups = [
                df[df[bin_column] == bin_label][pck_col].values
                for bin_label in unique_bins
                if not df[df[bin_column] == bin_label].empty
            ]

            if len(groups) > 1:
                f_stat, p_value = stats.f_oneway(*groups)
            else:
                f_stat, p_value = float("nan"), float("nan")

            result = {
                "PCK_Column": pck_col,
                "F_statistic": f_stat,
                "p_value": p_value,
                "Significant_at_0.05": p_value < 0.05 if pd.notna(p_value) else False,
            }
            results.append(result)

            print(f"\nPCK Column: {pck_col}")
            print(f"F-Statistic = {f_stat:.4f}, p-value = {p_value:.4f}")

            if p_value < 0.05:
                print("Significant difference found across bins (p < 0.05).")
            else:
                print("No significant difference found (p >= 0.05).")

        return results

    def _save_results(self, results: List[Dict[str, Any]], metric_name: str):
        """Save ANOVA results to Excel file."""
        results_df = pd.DataFrame(results)
        save_path = os.path.join(
            self.config.save_folder, f"{metric_name}_anova_results.xlsx"
        )

        os.makedirs(self.config.save_folder, exist_ok=True)

        if os.path.exists(save_path):
            existing_df = pd.read_excel(save_path)
            combined_df = pd.concat([existing_df, results_df], ignore_index=True)
            combined_df.to_excel(save_path, index=False)
            print(f"Updated ANOVA results saved to '{save_path}'")
        else:
            results_df.to_excel(save_path, index=False)
            print(f"ANOVA results created at '{save_path}'")


class BinAnalyzer(BaseAnalyzer):
    """Analyzer for bin-based statistical analysis."""

    def analyze(self, data: pd.DataFrame, metric_name: str) -> Dict[str, Any]:
        """Perform bin analysis on the data."""
        print(f"\nAnalyzing Metric: {metric_name.title()}")

        cleaned_data = self._clean_data(data, metric_name)
        binned_data = self._create_bins(cleaned_data, metric_name)
        frame_analysis = self._analyze_frames_per_bin(binned_data, metric_name)
        pck_stats = self._compute_pck_statistics(binned_data, metric_name)
        self._save_results(pck_stats, metric_name)

        return {"frame_analysis": frame_analysis, "pck_statistics": pck_stats}

    def _clean_data(self, df: pd.DataFrame, metric_name: str) -> pd.DataFrame:
        """Clean the data for analysis."""
        cleaned_df = df.copy()

        if cleaned_df.isnull().values.any():
            print("Warning: Missing values found. Filling with 0...")
            columns_to_fill = [metric_name] + self.config.pck_per_frame_score_columns
            cleaned_df.loc[:, columns_to_fill] = cleaned_df.loc[
                :, columns_to_fill
            ].fillna(0)
            cleaned_df.dropna(inplace=True)

        return cleaned_df

    def _create_bins(self, df: pd.DataFrame, metric_name: str) -> pd.DataFrame:
        """Create bins for the metric."""
        if metric_name == "brightness":
            bins = [0, 50, 100, 150, 200, 255]
            labels = ["Low", "Medium-Low", "Medium", "Medium-High", "High"]
        else:
            bins = pd.qcut(df[metric_name], q=5, retbins=True, labels=False)[1]
            labels = [f"Bin {i + 1}" for i in range(5)]

        df[f"{metric_name}_bin"] = pd.cut(
            df[metric_name], bins=bins, labels=labels, right=False
        )
        return df

    def _analyze_frames_per_bin(self, df: pd.DataFrame, metric_name: str) -> pd.Series:
        """Analyze number of frames per bin."""
        bin_column = f"{metric_name}_bin"
        frame_counts = df[bin_column].value_counts()

        print("\nNumber of frames per bin:")
        print(frame_counts)

        return frame_counts

    def _compute_pck_statistics(
        self, df: pd.DataFrame, metric_name: str
    ) -> pd.DataFrame:
        """Compute PCK statistics per bin."""
        bin_column = f"{metric_name}_bin"
        pck_summary_list = []

        for pck_col in self.config.pck_per_frame_score_columns:
            stats = (
                df.groupby(bin_column)[pck_col]
                .agg(mean_pck="mean", median_pck="median", std_pck="std", count="count")
                .reset_index()
            )
            stats["PCK_Column"] = pck_col
            pck_summary_list.append(stats)

        return pd.concat(pck_summary_list, ignore_index=True)

    def _save_results(self, summary_df: pd.DataFrame, metric_name: str):
        """Save bin analysis results."""
        save_path = os.path.join(self.config.save_folder, f"{metric_name}_summary.xlsx")

        os.makedirs(self.config.save_folder, exist_ok=True)

        if os.path.exists(save_path):
            existing_df = pd.read_excel(save_path)
            combined_df = pd.concat([existing_df, summary_df], ignore_index=True)
            combined_df.to_excel(save_path, index=False)
            print(f"Updated summary for {metric_name} saved to '{save_path}'")
        else:
            summary_df.to_excel(save_path, index=False)
            print(f"Summary for {metric_name} created at '{save_path}'")


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


class AnalyzerFactory:
    """Factory for creating analyzers."""

    _analyzers = {
        "anova": ANOVAAnalyzer,
        "bin_analysis": BinAnalyzer,
        "pck_frame_count": PCKFrameCountAnalyzer,
    }

    @classmethod
    def create_analyzer(cls, analyzer_type: str, config) -> BaseAnalyzer:
        """Create an analyzer of the specified type."""
        analyzer_type = analyzer_type.lower()

        if analyzer_type not in cls._analyzers:
            raise ValueError(
                f"Unknown analyzer type: {analyzer_type}. Available: {list(cls._analyzers.keys())}"
            )

        return cls._analyzers[analyzer_type](config)

    @classmethod
    def register_analyzer(cls, analyzer_type: str, analyzer_class: type):
        """Register a new analyzer type."""
        if not issubclass(analyzer_class, BaseAnalyzer):
            raise ValueError("Analyzer class must inherit from BaseAnalyzer")

        cls._analyzers[analyzer_type.lower()] = analyzer_class

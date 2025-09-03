"""
ANOVA analyzer for statistical testing.
"""

from ..base_classes import BaseAnalyzer
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

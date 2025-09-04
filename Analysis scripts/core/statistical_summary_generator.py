"""
Statistical Summary Generator Script

Generates statistical summaries for PCK brightness distributions.
Focus: Statistical analysis and summary generation only.
"""

import sys
import os
import pandas as pd
import numpy as np
from typing import List, Tuple
from scipy import stats

# Add the Analysis scripts directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from distribution_calculator import DistributionCalculator


class StatisticalSummaryGenerator:
    """Generate statistical summaries for PCK brightness distributions."""

    def __init__(self, dataset_name: str):
        """Initialize with dataset name."""
        self.dataset_name = dataset_name
        self.calculator = DistributionCalculator(dataset_name)

    def generate_descriptive_statistics(
        self, target_scores: List[int], pck_threshold: str = None, bin_size: int = 5
    ) -> pd.DataFrame:
        """Generate descriptive statistics for PCK scores."""
        print(f"Generating descriptive statistics for PCK scores: {target_scores}")

        # Calculate distributions
        distributions = self.calculator.calculate_distributions_for_scores(
            target_scores, pck_threshold, bin_size
        )

        if not distributions:
            print("❌ No distributions available for analysis")
            return pd.DataFrame()

        stats_data = []

        for score, dist in distributions.items():
            bin_centers = np.array(dist["bin_centers"])
            frequencies = np.array(dist["frequencies"])
            norm_frequencies = np.array(dist["normalized_frequencies"])

            # Create weighted data for statistics
            weighted_brightness = []
            for brightness, freq in zip(bin_centers, frequencies):
                weighted_brightness.extend([brightness] * freq)

            if len(weighted_brightness) == 0:
                continue

            weighted_brightness = np.array(weighted_brightness)

            # Calculate statistics
            stats_row = {
                "PCK_Score": score,
                "Total_Frames": dist["total_frames"],
                "Mean_Brightness": np.mean(weighted_brightness),
                "Median_Brightness": np.median(weighted_brightness),
                "Std_Brightness": np.std(weighted_brightness),
                "Min_Brightness": np.min(weighted_brightness),
                "Max_Brightness": np.max(weighted_brightness),
                "Q1_Brightness": np.percentile(weighted_brightness, 25),
                "Q3_Brightness": np.percentile(weighted_brightness, 75),
                "IQR_Brightness": np.percentile(weighted_brightness, 75)
                - np.percentile(weighted_brightness, 25),
                "Skewness": stats.skew(weighted_brightness),
                "Kurtosis": stats.kurtosis(weighted_brightness),
                "Range_Brightness": np.max(weighted_brightness)
                - np.min(weighted_brightness),
                "CV_Brightness": np.std(weighted_brightness)
                / np.mean(weighted_brightness)
                if np.mean(weighted_brightness) > 0
                else 0,
                "Peak_Brightness": bin_centers[np.argmax(norm_frequencies)],
                "Peak_Frequency": np.max(norm_frequencies),
            }

            stats_data.append(stats_row)

        df = pd.DataFrame(stats_data)

        if not df.empty:
            # Round numeric columns
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            df[numeric_columns] = df[numeric_columns].round(3)

            print(f"✅ Generated descriptive statistics for {len(df)} PCK scores")

        return df

    def generate_distribution_comparison(
        self, target_scores: List[int], pck_threshold: str = None, bin_size: int = 5
    ) -> pd.DataFrame:
        """Generate comparison metrics between distributions."""
        print(f"Generating distribution comparison for PCK scores: {target_scores}")

        distributions = self.calculator.calculate_distributions_for_scores(
            target_scores, pck_threshold, bin_size
        )

        if len(distributions) < 2:
            print("❌ Need at least 2 distributions for comparison")
            return pd.DataFrame()

        comparison_data = []
        scores = list(distributions.keys())

        for i in range(len(scores)):
            for j in range(i + 1, len(scores)):
                score1, score2 = scores[i], scores[j]
                dist1 = distributions[score1]
                dist2 = distributions[score2]

                # Get weighted data
                weighted1 = []
                for brightness, freq in zip(dist1["bin_centers"], dist1["frequencies"]):
                    weighted1.extend([brightness] * freq)

                weighted2 = []
                for brightness, freq in zip(dist2["bin_centers"], dist2["frequencies"]):
                    weighted2.extend([brightness] * freq)

                if len(weighted1) == 0 or len(weighted2) == 0:
                    continue

                # Statistical tests
                try:
                    # Kolmogorov-Smirnov test
                    ks_stat, ks_p = stats.ks_2samp(weighted1, weighted2)

                    # Mann-Whitney U test
                    mw_stat, mw_p = stats.mannwhitneyu(
                        weighted1, weighted2, alternative="two-sided"
                    )

                    # Welch's t-test (unequal variances)
                    t_stat, t_p = stats.ttest_ind(weighted1, weighted2, equal_var=False)

                except Exception as e:
                    print(
                        f"⚠️  Statistical test failed for PCK {score1} vs {score2}: {e}"
                    )
                    ks_stat = ks_p = mw_stat = mw_p = t_stat = t_p = np.nan

                # Effect size (Cohen's d)
                pooled_std = np.sqrt((np.var(weighted1) + np.var(weighted2)) / 2)
                cohens_d = (
                    (np.mean(weighted1) - np.mean(weighted2)) / pooled_std
                    if pooled_std > 0
                    else 0
                )

                # Mean difference
                mean_diff = np.mean(weighted1) - np.mean(weighted2)

                comparison_row = {
                    "PCK_Score_1": score1,
                    "PCK_Score_2": score2,
                    "Mean_Diff": mean_diff,
                    "Cohens_D": cohens_d,
                    "KS_Statistic": ks_stat,
                    "KS_P_Value": ks_p,
                    "KS_Significant": ks_p < 0.05 if not np.isnan(ks_p) else False,
                    "MW_Statistic": mw_stat,
                    "MW_P_Value": mw_p,
                    "MW_Significant": mw_p < 0.05 if not np.isnan(mw_p) else False,
                    "T_Statistic": t_stat,
                    "T_P_Value": t_p,
                    "T_Significant": t_p < 0.05 if not np.isnan(t_p) else False,
                    "Sample_Size_1": len(weighted1),
                    "Sample_Size_2": len(weighted2),
                }

                comparison_data.append(comparison_row)

        df = pd.DataFrame(comparison_data)

        if not df.empty:
            # Round numeric columns
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            df[numeric_columns] = df[numeric_columns].round(4)

            print(f"✅ Generated comparison for {len(df)} score pairs")

        return df

    def generate_correlation_analysis(
        self, target_scores: List[int], pck_threshold: str = None, bin_size: int = 5
    ) -> dict:
        """Generate correlation analysis between PCK scores and brightness metrics."""
        print(f"Generating correlation analysis for PCK scores: {target_scores}")

        distributions = self.calculator.calculate_distributions_for_scores(
            target_scores, pck_threshold, bin_size
        )

        if len(distributions) < 3:
            print("❌ Need at least 3 distributions for meaningful correlation")
            return {}

        # Collect metrics for correlation
        scores = []
        mean_brightness = []
        std_brightness = []
        peak_brightness = []
        total_frames = []

        for score, dist in distributions.items():
            # Calculate weighted mean brightness
            bin_centers = np.array(dist["bin_centers"])
            frequencies = np.array(dist["frequencies"])

            if sum(frequencies) == 0:
                continue

            weighted_mean = np.average(bin_centers, weights=frequencies)

            # Calculate weighted standard deviation
            weighted_var = np.average(
                (bin_centers - weighted_mean) ** 2, weights=frequencies
            )
            weighted_std = np.sqrt(weighted_var)

            # Peak brightness
            peak_idx = np.argmax(dist["normalized_frequencies"])
            peak_bright = bin_centers[peak_idx]

            scores.append(score)
            mean_brightness.append(weighted_mean)
            std_brightness.append(weighted_std)
            peak_brightness.append(peak_bright)
            total_frames.append(dist["total_frames"])

        if len(scores) < 3:
            print("❌ Insufficient valid data for correlation analysis")
            return {}

        # Calculate correlations
        correlations = {}

        try:
            # PCK Score vs Mean Brightness
            corr_mean, p_mean = stats.pearsonr(scores, mean_brightness)
            correlations["PCK_vs_Mean_Brightness"] = {
                "correlation": corr_mean,
                "p_value": p_mean,
                "significant": p_mean < 0.05,
            }

            # PCK Score vs Std Brightness
            corr_std, p_std = stats.pearsonr(scores, std_brightness)
            correlations["PCK_vs_Std_Brightness"] = {
                "correlation": corr_std,
                "p_value": p_std,
                "significant": p_std < 0.05,
            }

            # PCK Score vs Peak Brightness
            corr_peak, p_peak = stats.pearsonr(scores, peak_brightness)
            correlations["PCK_vs_Peak_Brightness"] = {
                "correlation": corr_peak,
                "p_value": p_peak,
                "significant": p_peak < 0.05,
            }

            # PCK Score vs Total Frames
            corr_frames, p_frames = stats.pearsonr(scores, total_frames)
            correlations["PCK_vs_Total_Frames"] = {
                "correlation": corr_frames,
                "p_value": p_frames,
                "significant": p_frames < 0.05,
            }

        except Exception as e:
            print(f"⚠️  Correlation calculation failed: {e}")
            return {}

        print(f"✅ Generated correlation analysis for {len(scores)} PCK scores")
        return correlations

    def generate_summary_report(
        self,
        target_scores: List[int],
        pck_threshold: str = None,
        bin_size: int = 5,
        save_report: bool = True,
        filename: str = None,
    ) -> Tuple[dict, str]:
        """Generate comprehensive statistical summary report."""
        print(
            f"Generating comprehensive summary report for PCK scores: {target_scores}"
        )

        report = {
            "dataset": self.dataset_name,
            "pck_threshold": pck_threshold,
            "bin_size": bin_size,
            "target_scores": target_scores,
            "descriptive_stats": None,
            "comparison_stats": None,
            "correlation_analysis": None,
            "summary_metrics": {},
        }

        # Generate components
        try:
            # Descriptive statistics
            desc_stats = self.generate_descriptive_statistics(
                target_scores, pck_threshold, bin_size
            )
            if not desc_stats.empty:
                report["descriptive_stats"] = desc_stats.to_dict("records")

            # Comparison statistics
            if len(target_scores) > 1:
                comp_stats = self.generate_distribution_comparison(
                    target_scores, pck_threshold, bin_size
                )
                if not comp_stats.empty:
                    report["comparison_stats"] = comp_stats.to_dict("records")

            # Correlation analysis
            if len(target_scores) > 2:
                corr_analysis = self.generate_correlation_analysis(
                    target_scores, pck_threshold, bin_size
                )
                if corr_analysis:
                    report["correlation_analysis"] = corr_analysis

            # Summary metrics
            if not desc_stats.empty:
                report["summary_metrics"] = {
                    "total_scores_analyzed": len(desc_stats),
                    "total_frames": desc_stats["Total_Frames"].sum(),
                    "overall_mean_brightness": desc_stats["Mean_Brightness"].mean(),
                    "brightness_range": {
                        "min": desc_stats["Min_Brightness"].min(),
                        "max": desc_stats["Max_Brightness"].max(),
                    },
                    "most_variable_score": int(
                        desc_stats.loc[
                            desc_stats["CV_Brightness"].idxmax(), "PCK_Score"
                        ]
                    ),
                    "least_variable_score": int(
                        desc_stats.loc[
                            desc_stats["CV_Brightness"].idxmin(), "PCK_Score"
                        ]
                    ),
                }

        except Exception as e:
            print(f"⚠️  Error generating report components: {e}")

        # Save report
        if save_report:
            if filename is None:
                scores_str = "_".join(map(str, target_scores))
                filename = (
                    f"statistical_summary_{self.dataset_name}_scores_{scores_str}.json"
                )

            from config import ConfigManager

            config = ConfigManager.load_config(self.dataset_name)

            import json

            output_path = os.path.join(config.save_folder, filename)
            os.makedirs(config.save_folder, exist_ok=True)

            with open(output_path, "w") as f:
                json.dump(report, f, indent=2, default=str)

            print(f"✅ Statistical summary report saved to: {output_path}")
            return report, output_path

        return report, ""

    def export_all_statistics_to_excel(
        self,
        target_scores: List[int],
        pck_threshold: str = None,
        bin_size: int = 5,
        filename: str = None,
    ) -> str:
        """Export all statistics to an Excel file with multiple sheets."""
        print(f"Exporting all statistics to Excel for PCK scores: {target_scores}")

        try:
            # Generate all statistics
            desc_stats = self.generate_descriptive_statistics(
                target_scores, pck_threshold, bin_size
            )

            comp_stats = (
                self.generate_distribution_comparison(
                    target_scores, pck_threshold, bin_size
                )
                if len(target_scores) > 1
                else pd.DataFrame()
            )

            # Prepare filename
            if filename is None:
                scores_str = "_".join(map(str, target_scores))
                filename = (
                    f"all_statistics_{self.dataset_name}_scores_{scores_str}.xlsx"
                )

            from config import ConfigManager

            config = ConfigManager.load_config(self.dataset_name)
            output_path = os.path.join(config.save_folder, filename)
            os.makedirs(config.save_folder, exist_ok=True)

            # Write to Excel with multiple sheets
            with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
                if not desc_stats.empty:
                    desc_stats.to_excel(
                        writer, sheet_name="Descriptive_Stats", index=False
                    )

                if not comp_stats.empty:
                    comp_stats.to_excel(
                        writer, sheet_name="Comparison_Stats", index=False
                    )

                # Add metadata sheet
                metadata = pd.DataFrame(
                    {
                        "Parameter": [
                            "Dataset",
                            "PCK Threshold",
                            "Bin Size",
                            "Target Scores",
                            "Generated On",
                        ],
                        "Value": [
                            self.dataset_name,
                            pck_threshold or "All",
                            bin_size,
                            ", ".join(map(str, target_scores)),
                            pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                        ],
                    }
                )
                metadata.to_excel(writer, sheet_name="Metadata", index=False)

            print(f"✅ All statistics exported to Excel: {output_path}")
            return output_path

        except Exception as e:
            print(f"❌ Failed to export to Excel: {e}")
            return ""


def main():
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Statistical Summary Generator")
    parser.add_argument("dataset", help="Dataset name (e.g., 'movi', 'humaneva')")
    parser.add_argument("scores", nargs="+", type=int, help="PCK scores to analyze")
    parser.add_argument("--threshold", help="Specific PCK threshold to analyze")
    parser.add_argument("--bin-size", type=int, default=5, help="Brightness bin size")
    parser.add_argument(
        "--type",
        choices=["descriptive", "comparison", "correlation", "report", "excel"],
        default="descriptive",
        help="Type of statistical analysis",
    )
    parser.add_argument("--filename", help="Custom filename for output")

    args = parser.parse_args()

    try:
        generator = StatisticalSummaryGenerator(args.dataset)

        if args.type == "descriptive":
            stats_df = generator.generate_descriptive_statistics(
                args.scores, args.threshold, args.bin_size
            )
            if not stats_df.empty:
                print("\n" + "=" * 50)
                print("DESCRIPTIVE STATISTICS")
                print("=" * 50)
                print(stats_df.to_string(index=False))

        elif args.type == "comparison":
            comp_df = generator.generate_distribution_comparison(
                args.scores, args.threshold, args.bin_size
            )
            if not comp_df.empty:
                print("\n" + "=" * 50)
                print("DISTRIBUTION COMPARISON")
                print("=" * 50)
                print(comp_df.to_string(index=False))

        elif args.type == "correlation":
            corr_analysis = generator.generate_correlation_analysis(
                args.scores, args.threshold, args.bin_size
            )
            if corr_analysis:
                print("\n" + "=" * 50)
                print("CORRELATION ANALYSIS")
                print("=" * 50)
                for key, value in corr_analysis.items():
                    print(
                        f"{key}: r={value['correlation']:.4f}, p={value['p_value']:.4f}, sig={value['significant']}"
                    )

        elif args.type == "report":
            report, path = generator.generate_summary_report(
                args.scores, args.threshold, args.bin_size, True, args.filename
            )
            print(f"Summary report generated with {len(report)} sections")

        elif args.type == "excel":
            path = generator.export_all_statistics_to_excel(
                args.scores, args.threshold, args.bin_size, args.filename
            )
            if path:
                print(f"Excel export completed: {path}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

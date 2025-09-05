"""
Joint Analysis Report Generator

Handles generation of Excel summaries and text reports for joint analysis.
"""

import pandas as pd
from typing import Dict, Any, List
from pathlib import Path
from datetime import datetime


class JointReportGenerator:
    """Handles report generation for joint analysis results."""

    def __init__(self, output_dir: Path, dataset_name: str, save_reports: bool = True):
        """Initialize the report generator.

        Args:
            output_dir: Directory to save reports
            dataset_name: Name of the dataset being analyzed
            save_reports: Whether to save reports to files
        """
        self.output_dir = output_dir
        self.dataset_name = dataset_name
        self.save_reports = save_reports

        if save_reports:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_excel_summary(
        self,
        analysis_results: Dict[str, Any],
        joints_to_analyze: List[str],
        pck_thresholds: List[float],
    ) -> None:
        """Generate Excel summary files for each threshold.

        Args:
            analysis_results: Complete analysis results
            joints_to_analyze: List of joints analyzed
            pck_thresholds: List of PCK thresholds analyzed
        """
        if not analysis_results:
            print("ERROR: No analysis results to generate Excel summaries")
            return

        print("Generating Excel summary files...")

        try:
            for threshold in pck_thresholds:
                print(f"  Creating Excel summary for threshold {threshold}...")

                summary_data = []

                for joint_name in joints_to_analyze:
                    metric_key = f"{joint_name}_jointwise_pck_{threshold:g}"

                    if metric_key not in analysis_results:
                        continue

                    result = analysis_results[metric_key]
                    brightness_stats = result["brightness_stats"]

                    summary_data.append(
                        {
                            "Joint_Name": joint_name.replace("_", " ").title(),
                            "Average_PCK_Score": round(result["avg_pck"], 4),
                            "Mean_Brightness": round(brightness_stats["mean"], 2),
                            "Std_Deviation": round(brightness_stats["std"], 2),
                            "IQR": round(brightness_stats["iqr"], 2),
                            "Frame_Count": brightness_stats["count"],
                        }
                    )

                if summary_data and self.save_reports:
                    df = pd.DataFrame(summary_data)

                    # Add overall summary row
                    if len(summary_data) > 1:
                        overall_row = {
                            "Joint_Name": "OVERALL AVERAGE",
                            "Average_PCK_Score": round(
                                df["Average_PCK_Score"].mean(), 4
                            ),
                            "Mean_Brightness": round(df["Mean_Brightness"].mean(), 2),
                            "Std_Deviation": round(df["Std_Deviation"].mean(), 2),
                            "IQR": round(df["IQR"].mean(), 2),
                            "Frame_Count": df["Frame_Count"].sum(),
                        }
                        # Add a separator row and overall row
                        separator_row = {col: "---" for col in df.columns}
                        df = pd.concat(
                            [df, pd.DataFrame([separator_row, overall_row])],
                            ignore_index=True,
                        )

                    excel_file = (
                        self.output_dir
                        / f"brightness_summary_threshold_{threshold:g}.xlsx"
                    )

                    # Save to Excel with formatting
                    with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:
                        df.to_excel(
                            writer, sheet_name="Brightness_Summary", index=False
                        )

                        # Format the worksheet
                        worksheet = writer.sheets["Brightness_Summary"]

                        # Adjust column widths
                        for column in worksheet.columns:
                            max_length = 0
                            column = [cell for cell in column]
                            for cell in column:
                                try:
                                    if len(str(cell.value)) > max_length:
                                        max_length = len(str(cell.value))
                                except Exception:
                                    pass
                            adjusted_width = max_length + 2
                            worksheet.column_dimensions[
                                column[0].column_letter
                            ].width = adjusted_width

                    print(f"     Saved Excel summary: {excel_file}")

                    # Print summary to console
                    print(f"\\n     Summary for threshold {threshold}:")
                    print("     " + "=" * 50)
                    print(df.to_string(index=False))
                    print()

            print("Excel summary generation completed")

        except Exception as e:
            print(f"ERROR: Error generating Excel summaries: {e}")
            import traceback

            traceback.print_exc()

    def generate_text_report(
        self,
        analysis_results: Dict[str, Any],
        joints_to_analyze: List[str],
        pck_thresholds: List[float],
    ) -> None:
        """Generate text analysis report.

        Args:
            analysis_results: Complete analysis results
            joints_to_analyze: List of joints analyzed
            pck_thresholds: List of PCK thresholds analyzed
        """
        if not self.save_reports:
            return

        try:
            report_file = self.output_dir / "joint_analysis_report.txt"

            with open(report_file, "w") as f:
                f.write("Joint Analysis Report\\n")
                f.write("=" * 50 + "\\n\\n")

                f.write(f"Dataset: {self.dataset_name}\\n")
                f.write(
                    f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n"
                )
                f.write(f"Joints Analyzed: {', '.join(joints_to_analyze)}\\n")
                f.write(f"PCK Thresholds: {pck_thresholds}\\n\\n")

                f.write("Analysis Results Summary:\\n")
                f.write("-" * 30 + "\\n")
                f.write(f"Total Metrics: {len(analysis_results)}\\n\\n")

                # Summary statistics for each metric
                for metric_name, result in analysis_results.items():
                    f.write(f"{metric_name}:\\n")
                    f.write(f"  Average PCK: {result['avg_pck']:.4f}\\n")
                    f.write(f"  Average Brightness: {result['avg_brightness']:.2f}\\n")
                    f.write(f"  Correlation: {result['correlation']:.4f}\\n")
                    f.write(
                        f"  Sample Count: {result['brightness_stats']['count']}\\n\\n"
                    )

                # Correlation analysis
                f.write("\\nCorrelation Analysis:\\n")
                f.write("-" * 20 + "\\n")
                for threshold in pck_thresholds:
                    f.write(f"\\nThreshold {threshold}:\\n")
                    for joint_name in joints_to_analyze:
                        metric_key = f"{joint_name}_jointwise_pck_{threshold:g}"
                        if metric_key in analysis_results:
                            corr = analysis_results[metric_key]["correlation"]
                            f.write(f"  {joint_name}: {corr:.4f}\\n")

            print(f"Text report saved to: {report_file}")

        except Exception as e:
            print(f"WARNING: Could not save text report: {e}")

    def generate_consolidated_report(
        self,
        analysis_results: Dict[str, Any],
        joints_to_analyze: List[str],
        pck_thresholds: List[float],
    ) -> None:
        """Generate consolidated report with all thresholds and metrics in one Excel file.

        Args:
            analysis_results: Complete analysis results
            joints_to_analyze: List of joints analyzed
            pck_thresholds: List of PCK thresholds analyzed
        """
        if not analysis_results or not self.save_reports:
            return

        print("Generating consolidated report with all thresholds...")

        try:
            consolidated_data = []

            for joint_name in joints_to_analyze:
                joint_row = {
                    "Joint_Name": joint_name.replace("_", " ").title(),
                }

                # Add metrics for each threshold
                for threshold in pck_thresholds:
                    metric_key = f"{joint_name}_jointwise_pck_{threshold:g}"

                    if metric_key in analysis_results:
                        result = analysis_results[metric_key]
                        brightness_stats = result["brightness_stats"]

                        # Add PCK score for this threshold
                        joint_row[f"PCK_Score_{threshold:g}"] = round(
                            result["avg_pck"], 4
                        )

                        # Add brightness metrics for this threshold
                        joint_row[f"Mean_Brightness_{threshold:g}"] = round(
                            brightness_stats["mean"], 2
                        )
                        joint_row[f"Std_Brightness_{threshold:g}"] = round(
                            brightness_stats["std"], 2
                        )
                        joint_row[f"IQR_Brightness_{threshold:g}"] = round(
                            brightness_stats["iqr"], 2
                        )
                        joint_row[f"Correlation_{threshold:g}"] = round(
                            result["correlation"], 4
                        )
                        joint_row[f"Frame_Count_{threshold:g}"] = brightness_stats[
                            "count"
                        ]
                    else:
                        # Fill with None if data not available
                        joint_row[f"PCK_Score_{threshold:g}"] = None
                        joint_row[f"Mean_Brightness_{threshold:g}"] = None
                        joint_row[f"Std_Brightness_{threshold:g}"] = None
                        joint_row[f"IQR_Brightness_{threshold:g}"] = None
                        joint_row[f"Correlation_{threshold:g}"] = None
                        joint_row[f"Frame_Count_{threshold:g}"] = None

                consolidated_data.append(joint_row)

            if consolidated_data:
                df = pd.DataFrame(consolidated_data)

                # Add overall summary row
                if len(consolidated_data) > 1:
                    overall_row = {"Joint_Name": "OVERALL AVERAGE"}

                    for threshold in pck_thresholds:
                        # Calculate averages for each threshold
                        pck_col = f"PCK_Score_{threshold:g}"
                        brightness_col = f"Mean_Brightness_{threshold:g}"
                        std_col = f"Std_Brightness_{threshold:g}"
                        iqr_col = f"IQR_Brightness_{threshold:g}"
                        corr_col = f"Correlation_{threshold:g}"
                        count_col = f"Frame_Count_{threshold:g}"

                        if pck_col in df.columns:
                            overall_row[pck_col] = round(
                                df[pck_col].mean(skipna=True), 4
                            )
                        if brightness_col in df.columns:
                            overall_row[brightness_col] = round(
                                df[brightness_col].mean(skipna=True), 2
                            )
                        if std_col in df.columns:
                            overall_row[std_col] = round(
                                df[std_col].mean(skipna=True), 2
                            )
                        if iqr_col in df.columns:
                            overall_row[iqr_col] = round(
                                df[iqr_col].mean(skipna=True), 2
                            )
                        if corr_col in df.columns:
                            overall_row[corr_col] = round(
                                df[corr_col].mean(skipna=True), 4
                            )
                        if count_col in df.columns:
                            overall_row[count_col] = df[count_col].sum(skipna=True)

                    # Add separator and overall row
                    separator_row = {col: "---" for col in df.columns}
                    df = pd.concat(
                        [df, pd.DataFrame([separator_row, overall_row])],
                        ignore_index=True,
                    )

                # Save consolidated Excel file
                excel_file = self.output_dir / "consolidated_analysis_report.xlsx"

                with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:
                    # Main consolidated sheet
                    df.to_excel(writer, sheet_name="Consolidated_Analysis", index=False)

                    # Format the main worksheet
                    worksheet = writer.sheets["Consolidated_Analysis"]

                    # Auto-adjust column widths
                    for column in worksheet.columns:
                        max_length = 0
                        column = [cell for cell in column]
                        for cell in column:
                            try:
                                if len(str(cell.value)) > max_length:
                                    max_length = len(str(cell.value))
                            except Exception:
                                pass
                        adjusted_width = min(max_length + 2, 20)  # Cap width at 20
                        worksheet.column_dimensions[
                            column[0].column_letter
                        ].width = adjusted_width

                    # Create summary sheet with overview statistics
                    summary_data = []
                    for threshold in pck_thresholds:
                        threshold_summary = {
                            "Threshold": threshold,
                            "Avg_PCK_Score": round(
                                df[f"PCK_Score_{threshold:g}"]
                                .iloc[:-2]
                                .mean(skipna=True),
                                4,
                            ),
                            "Avg_Brightness": round(
                                df[f"Mean_Brightness_{threshold:g}"]
                                .iloc[:-2]
                                .mean(skipna=True),
                                2,
                            ),
                            "Avg_Correlation": round(
                                df[f"Correlation_{threshold:g}"]
                                .iloc[:-2]
                                .mean(skipna=True),
                                4,
                            ),
                            "Total_Frames": df[f"Frame_Count_{threshold:g}"]
                            .iloc[:-2]
                            .sum(skipna=True),
                            "Joints_Analyzed": df[f"PCK_Score_{threshold:g}"]
                            .iloc[:-2]
                            .notna()
                            .sum(),
                        }
                        summary_data.append(threshold_summary)

                    summary_df = pd.DataFrame(summary_data)
                    summary_df.to_excel(
                        writer, sheet_name="Summary_by_Threshold", index=False
                    )

                    # Format summary worksheet
                    summary_ws = writer.sheets["Summary_by_Threshold"]
                    for column in summary_ws.columns:
                        max_length = 0
                        column = [cell for cell in column]
                        for cell in column:
                            try:
                                if len(str(cell.value)) > max_length:
                                    max_length = len(str(cell.value))
                            except Exception:
                                pass
                        adjusted_width = max_length + 2
                        summary_ws.column_dimensions[
                            column[0].column_letter
                        ].width = adjusted_width

                print("     Saved consolidated report: {excel_file}")

                # Print summary to console
                print("\n     Consolidated Analysis Summary:")
                print("     " + "=" * 60)
                print(summary_df.to_string(index=False))
                print()

        except Exception as e:
            print(f"ERROR: Error generating consolidated report: {e}")
            import traceback

            traceback.print_exc()

    def generate_all_reports(
        self,
        analysis_results: Dict[str, Any],
        joints_to_analyze: List[str],
        pck_thresholds: List[float],
    ) -> None:
        """Generate all reports (Excel and text).

        Args:
            analysis_results: Complete analysis results
            joints_to_analyze: List of joints analyzed
            pck_thresholds: List of PCK thresholds analyzed
        """
        print("Generating analysis reports...")

        # Generate Excel summaries
        self.generate_excel_summary(analysis_results, joints_to_analyze, pck_thresholds)

        # Generate consolidated report
        self.generate_consolidated_report(
            analysis_results, joints_to_analyze, pck_thresholds
        )

        # Generate text report
        self.generate_text_report(analysis_results, joints_to_analyze, pck_thresholds)

        print("Report generation completed")

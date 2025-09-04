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
                    metric_key = f"{joint_name}_pck_{threshold:g}"

                    if metric_key not in analysis_results:
                        continue

                    result = analysis_results[metric_key]
                    brightness_stats = result["brightness_stats"]

                    summary_data.append(
                        {
                            "Joint_Name": joint_name.replace("_", " ").title(),
                            "Mean_Brightness": round(brightness_stats["mean"], 2),
                            "Std_Deviation": round(brightness_stats["std"], 2),
                            "IQR": round(brightness_stats["iqr"], 2),
                            "Frame_Count": brightness_stats["count"],
                        }
                    )

                if summary_data and self.save_reports:
                    df = pd.DataFrame(summary_data)

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
                        metric_key = f"{joint_name}_pck_{threshold:g}"
                        if metric_key in analysis_results:
                            corr = analysis_results[metric_key]["correlation"]
                            f.write(f"  {joint_name}: {corr:.4f}\\n")

            print(f"Text report saved to: {report_file}")

        except Exception as e:
            print(f"WARNING: Could not save text report: {e}")

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

        # Generate text report
        self.generate_text_report(analysis_results, joints_to_analyze, pck_thresholds)

        print("Report generation completed")

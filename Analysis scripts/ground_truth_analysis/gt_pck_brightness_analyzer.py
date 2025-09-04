"""
Ground Truth PCK Brightness Analyzer Script

Analyzes the relationship between PCK scores and brightness at ground truth joint coordinates.
Focus: PCK-brightness correlation analysis using ground truth data.
"""

import sys
import os
import pandas as pd
import numpy as np
from typing import List, Dict

# Add the Analysis scripts directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gt_data_loader import GroundTruthDataLoader
from joint_analysis.joint_brightness_extractor import JointBrightnessExtractor
from simple_analysis.pck_loader import PCKDataLoader


class GTPCKBrightnessAnalyzer:
    """Analyze PCK scores vs brightness at ground truth joint coordinates."""

    def __init__(self, dataset_name: str):
        """Initialize with dataset name."""
        self.dataset_name = dataset_name
        self.gt_loader = GroundTruthDataLoader(dataset_name)
        self.brightness_extractor = JointBrightnessExtractor(dataset_name)
        self.pck_loader = PCKDataLoader(dataset_name)

    def analyze_pck_brightness_correlation(
        self,
        joint_names: List[str] = None,
        pck_threshold: str = None,
        video_path: str = None,
        sampling_radius: int = 3,
        **gt_filter_kwargs,
    ) -> Dict:
        """Analyze correlation between PCK scores and brightness at GT coordinates."""
        print(f"Analyzing PCK-brightness correlation for joints: {joint_names}")

        results = {
            "dataset": self.dataset_name,
            "joint_names": joint_names,
            "pck_threshold": pck_threshold,
            "correlations": {},
            "brightness_data": {},
            "pck_data": {},
            "summary_stats": {},
        }

        try:
            # 1. Extract joint brightness from ground truth coordinates
            print("\n📍 Step 1: Extracting brightness at GT joint coordinates...")
            brightness_data = self.brightness_extractor.extract_brightness_for_joints(
                joint_names=joint_names,
                video_path=video_path,
                sampling_radius=sampling_radius,
                **gt_filter_kwargs,
            )

            if not brightness_data:
                print("❌ No brightness data extracted")
                return results

            results["brightness_data"] = brightness_data

            # 2. Load PCK scores
            print("\n📊 Step 2: Loading PCK scores...")
            per_frame_data = self.pck_loader.load_per_frame_data()

            if per_frame_data is None:
                print("❌ No PCK data available")
                return results

            # Get PCK columns to analyze
            from config import ConfigManager

            config = ConfigManager.load_config(self.dataset_name)

            pck_columns = (
                [pck_threshold] if pck_threshold else config.pck_per_frame_score_columns
            )

            # 3. Align PCK data with brightness data
            print("\n🔗 Step 3: Aligning PCK and brightness data...")
            aligned_data = self._align_pck_brightness_data(
                brightness_data, per_frame_data, pck_columns
            )

            if not aligned_data:
                print("❌ No aligned data available")
                return results

            results["pck_data"] = aligned_data

            # 4. Calculate correlations
            print("\n📈 Step 4: Calculating correlations...")
            correlations = self._calculate_correlations(aligned_data)
            results["correlations"] = correlations

            # 5. Generate summary statistics
            print("\n📋 Step 5: Generating summary statistics...")
            summary_stats = self._generate_summary_statistics(aligned_data)
            results["summary_stats"] = summary_stats

            print("✅ PCK-brightness correlation analysis completed")
            return results

        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            import traceback

            traceback.print_exc()
            return results

    def _align_pck_brightness_data(
        self,
        brightness_data: Dict[str, List[float]],
        pck_data: pd.DataFrame,
        pck_columns: List[str],
    ) -> Dict:
        """Align brightness and PCK data by frame."""
        aligned_data = {}

        # Get minimum frame count across all data
        min_frames = min(len(values) for values in brightness_data.values())
        min_frames = min(min_frames, len(pck_data))

        print(f"   Aligning data for {min_frames} frames")

        for joint_name, brightness_values in brightness_data.items():
            joint_data = {
                "frames": list(range(min_frames)),
                "brightness": brightness_values[:min_frames],
                "pck_scores": {},
            }

            # Add PCK scores for each threshold
            for pck_col in pck_columns:
                if pck_col in pck_data.columns:
                    pck_scores = pck_data[pck_col].iloc[:min_frames].round().astype(int)
                    joint_data["pck_scores"][pck_col] = pck_scores.tolist()
                else:
                    print(f"⚠️  PCK column {pck_col} not found")

            aligned_data[joint_name] = joint_data

        return aligned_data

    def _calculate_correlations(self, aligned_data: Dict) -> Dict:
        """Calculate correlations between PCK scores and brightness."""
        correlations = {}

        for joint_name, joint_data in aligned_data.items():
            joint_correlations = {}
            brightness_values = joint_data["brightness"]

            # Remove NaN values for correlation calculation
            valid_indices = [
                i for i, b in enumerate(brightness_values) if not np.isnan(b)
            ]
            valid_brightness = [brightness_values[i] for i in valid_indices]

            if len(valid_brightness) < 3:
                print(f"⚠️  Insufficient valid data for {joint_name}")
                continue

            for pck_threshold, pck_scores in joint_data["pck_scores"].items():
                valid_pck = [pck_scores[i] for i in valid_indices]

                if len(valid_pck) == len(valid_brightness):
                    try:
                        # Pearson correlation
                        correlation = np.corrcoef(valid_brightness, valid_pck)[0, 1]

                        # Additional statistics
                        brightness_mean = np.mean(valid_brightness)
                        pck_mean = np.mean(valid_pck)

                        joint_correlations[pck_threshold] = {
                            "correlation": correlation,
                            "valid_samples": len(valid_brightness),
                            "brightness_mean": brightness_mean,
                            "pck_mean": pck_mean,
                            "brightness_std": np.std(valid_brightness),
                            "pck_std": np.std(valid_pck),
                        }

                    except Exception as e:
                        print(
                            f"⚠️  Correlation calculation failed for {joint_name}-{pck_threshold}: {e}"
                        )

            correlations[joint_name] = joint_correlations

        return correlations

    def _generate_summary_statistics(self, aligned_data: Dict) -> Dict:
        """Generate summary statistics for the analysis."""
        summary = {
            "total_joints": len(aligned_data),
            "joint_stats": {},
            "overall_brightness": {},
            "overall_pck": {},
        }

        all_brightness = []
        all_pck_scores = {}

        for joint_name, joint_data in aligned_data.items():
            brightness_values = [b for b in joint_data["brightness"] if not np.isnan(b)]

            if brightness_values:
                joint_stats = {
                    "valid_frames": len(brightness_values),
                    "brightness_mean": np.mean(brightness_values),
                    "brightness_std": np.std(brightness_values),
                    "brightness_range": [
                        np.min(brightness_values),
                        np.max(brightness_values),
                    ],
                }

                summary["joint_stats"][joint_name] = joint_stats
                all_brightness.extend(brightness_values)

            # Collect PCK scores
            for pck_threshold, pck_scores in joint_data["pck_scores"].items():
                if pck_threshold not in all_pck_scores:
                    all_pck_scores[pck_threshold] = []
                all_pck_scores[pck_threshold].extend(pck_scores)

        # Overall statistics
        if all_brightness:
            summary["overall_brightness"] = {
                "mean": np.mean(all_brightness),
                "std": np.std(all_brightness),
                "range": [np.min(all_brightness), np.max(all_brightness)],
            }

        for pck_threshold, scores in all_pck_scores.items():
            if scores:
                summary["overall_pck"][pck_threshold] = {
                    "mean": np.mean(scores),
                    "std": np.std(scores),
                    "unique_scores": sorted(list(set(scores))),
                }

        return summary

    def generate_correlation_report(self, analysis_results: Dict) -> pd.DataFrame:
        """Generate a correlation report DataFrame."""
        if "correlations" not in analysis_results:
            return pd.DataFrame()

        report_data = []

        for joint_name, joint_correlations in analysis_results["correlations"].items():
            for pck_threshold, corr_data in joint_correlations.items():
                row = {
                    "joint": joint_name,
                    "pck_threshold": pck_threshold,
                    "correlation": corr_data["correlation"],
                    "valid_samples": corr_data["valid_samples"],
                    "brightness_mean": corr_data["brightness_mean"],
                    "pck_mean": corr_data["pck_mean"],
                    "brightness_std": corr_data["brightness_std"],
                    "pck_std": corr_data["pck_std"],
                    "strong_correlation": abs(corr_data["correlation"]) > 0.5,
                    "correlation_direction": "positive"
                    if corr_data["correlation"] > 0
                    else "negative",
                }
                report_data.append(row)

        report_df = pd.DataFrame(report_data)

        if not report_df.empty:
            # Round numeric columns
            numeric_columns = report_df.select_dtypes(include=[np.number]).columns
            report_df[numeric_columns] = report_df[numeric_columns].round(4)

            # Sort by absolute correlation strength
            report_df = report_df.sort_values("correlation", key=abs, ascending=False)

        return report_df

    def analyze_brightness_by_pck_score(
        self, analysis_results: Dict, target_pck_scores: List[int] = None
    ) -> Dict:
        """Analyze brightness distributions for specific PCK scores."""
        if "pck_data" not in analysis_results:
            return {}

        brightness_by_pck = {}

        for joint_name, joint_data in analysis_results["pck_data"].items():
            joint_brightness_by_pck = {}
            brightness_values = joint_data["brightness"]

            for pck_threshold, pck_scores in joint_data["pck_scores"].items():
                pck_brightness_map = {}

                for frame_idx, (brightness, pck_score) in enumerate(
                    zip(brightness_values, pck_scores)
                ):
                    if not np.isnan(brightness):
                        if pck_score not in pck_brightness_map:
                            pck_brightness_map[pck_score] = []
                        pck_brightness_map[pck_score].append(brightness)

                # Filter by target scores if specified
                if target_pck_scores:
                    pck_brightness_map = {
                        score: values
                        for score, values in pck_brightness_map.items()
                        if score in target_pck_scores
                    }

                # Calculate statistics for each PCK score
                pck_brightness_stats = {}
                for pck_score, brightness_list in pck_brightness_map.items():
                    if brightness_list:
                        pck_brightness_stats[pck_score] = {
                            "frame_count": len(brightness_list),
                            "mean_brightness": np.mean(brightness_list),
                            "std_brightness": np.std(brightness_list),
                            "median_brightness": np.median(brightness_list),
                            "brightness_range": [
                                np.min(brightness_list),
                                np.max(brightness_list),
                            ],
                        }

                joint_brightness_by_pck[pck_threshold] = pck_brightness_stats

            brightness_by_pck[joint_name] = joint_brightness_by_pck

        return brightness_by_pck

    def export_analysis_results(
        self, analysis_results: Dict, output_filename: str = None
    ) -> str:
        """Export analysis results to JSON and CSV files."""
        try:
            if output_filename is None:
                joints_str = "_".join(analysis_results.get("joint_names", ["all"])[:3])
                threshold_str = analysis_results.get("pck_threshold", "all_thresholds")
                output_filename = f"gt_pck_brightness_analysis_{self.dataset_name}_{joints_str}_{threshold_str}"

            from config import ConfigManager

            config = ConfigManager.load_config(self.dataset_name)
            output_dir = config.save_folder
            os.makedirs(output_dir, exist_ok=True)

            # Export JSON summary
            import json

            json_path = os.path.join(output_dir, f"{output_filename}.json")

            # Prepare JSON-serializable data
            json_data = {
                "dataset": analysis_results["dataset"],
                "joint_names": analysis_results["joint_names"],
                "pck_threshold": analysis_results["pck_threshold"],
                "correlations": analysis_results["correlations"],
                "summary_stats": analysis_results["summary_stats"],
            }

            with open(json_path, "w") as f:
                json.dump(json_data, f, indent=2, default=str)

            print(f"✅ Exported JSON analysis to: {json_path}")

            # Export correlation report CSV
            correlation_df = self.generate_correlation_report(analysis_results)
            if not correlation_df.empty:
                csv_path = os.path.join(
                    output_dir, f"{output_filename}_correlations.csv"
                )
                correlation_df.to_csv(csv_path, index=False)
                print(f"✅ Exported correlation CSV to: {csv_path}")

            return json_path

        except Exception as e:
            print(f"❌ Failed to export analysis results: {e}")
            return ""


def main():
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Ground Truth PCK Brightness Analyzer")
    parser.add_argument("dataset", help="Dataset name (e.g., 'humaneva', 'movi')")
    parser.add_argument(
        "--joints", nargs="*", help="Specific joints to analyze (default: all)"
    )
    parser.add_argument("--pck-threshold", help="Specific PCK threshold to analyze")
    parser.add_argument("--video", help="Path to specific video file")
    parser.add_argument("--subject", help="Filter ground truth by subject")
    parser.add_argument("--action", help="Filter ground truth by action")
    parser.add_argument("--camera", type=int, help="Filter ground truth by camera")
    parser.add_argument(
        "--radius", type=int, default=3, help="Sampling radius around joint"
    )
    parser.add_argument("--export", action="store_true", help="Export analysis results")
    parser.add_argument("--filename", help="Custom output filename")
    parser.add_argument(
        "--target-scores", nargs="*", type=int, help="Analyze specific PCK scores"
    )

    args = parser.parse_args()

    try:
        analyzer = GTPCKBrightnessAnalyzer(args.dataset)

        # Set up ground truth filters
        gt_filters = {}
        if args.subject:
            gt_filters["subject"] = args.subject
        if args.action:
            gt_filters["action"] = args.action
        if args.camera is not None:
            gt_filters["camera"] = args.camera

        # Run analysis
        analysis_results = analyzer.analyze_pck_brightness_correlation(
            joint_names=args.joints,
            pck_threshold=args.pck_threshold,
            video_path=args.video,
            sampling_radius=args.radius,
            **gt_filters,
        )

        if not analysis_results.get("correlations"):
            print("❌ No correlation results available")
            return

        # Display correlation results
        correlation_df = analyzer.generate_correlation_report(analysis_results)
        if not correlation_df.empty:
            print("\n" + "=" * 80)
            print("PCK-BRIGHTNESS CORRELATION ANALYSIS")
            print("=" * 80)
            print(correlation_df.to_string(index=False))

        # Analyze brightness by PCK score if requested
        if args.target_scores:
            brightness_by_pck = analyzer.analyze_brightness_by_pck_score(
                analysis_results, args.target_scores
            )

            print("\n" + "=" * 80)
            print(f"BRIGHTNESS ANALYSIS FOR PCK SCORES: {args.target_scores}")
            print("=" * 80)

            for joint_name, joint_data in brightness_by_pck.items():
                print(f"\n{joint_name}:")
                for pck_threshold, pck_data in joint_data.items():
                    print(f"  {pck_threshold}:")
                    for pck_score, stats in pck_data.items():
                        print(
                            f"    PCK {pck_score}: {stats['frame_count']} frames, "
                            f"mean brightness: {stats['mean_brightness']:.2f}"
                        )

        # Export results
        if args.export:
            analyzer.export_analysis_results(analysis_results, args.filename)

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

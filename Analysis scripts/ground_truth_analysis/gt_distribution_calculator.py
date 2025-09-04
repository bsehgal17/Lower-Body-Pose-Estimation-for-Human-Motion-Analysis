"""
Ground Truth Distribution Calculator Script

Calculates brightness distributions and PCK score distributions from ground truth analysis.
Focus: Statistical distribution calculations for GT data only.
"""

import sys
import os
import numpy as np
import pandas as pd
from typing import List, Dict

# Add the Analysis scripts directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gt_pck_brightness_analyzer import GTPCKBrightnessAnalyzer


class GTDistributionCalculator:
    """Calculate distributions for ground truth PCK-brightness analysis."""

    def __init__(self, dataset_name: str):
        """Initialize with dataset name."""
        self.dataset_name = dataset_name
        self.analyzer = GTPCKBrightnessAnalyzer(dataset_name)

    def calculate_brightness_distributions_by_pck(
        self, analysis_results: Dict, bin_size: int = 2, joint_name: str = None
    ) -> Dict:
        """Calculate brightness distributions for each PCK score."""
        if "pck_data" not in analysis_results:
            print("❌ No PCK data in analysis results")
            return {}

        print(f"Calculating brightness distributions with bin size: {bin_size}")

        # Use first joint if none specified
        if joint_name is None:
            joint_name = list(analysis_results["pck_data"].keys())[0]
            print(f"Using joint: {joint_name}")

        if joint_name not in analysis_results["pck_data"]:
            print(f"❌ Joint {joint_name} not found in analysis results")
            return {}

        joint_data = analysis_results["pck_data"][joint_name]
        distributions = {}

        for pck_threshold, pck_scores in joint_data["pck_scores"].items():
            print(f"   Processing PCK threshold: {pck_threshold}")

            # Group brightness values by PCK score
            pck_brightness_map = {}
            brightness_values = joint_data["brightness"]

            for brightness, pck_score in zip(brightness_values, pck_scores):
                if not np.isnan(brightness):
                    if pck_score not in pck_brightness_map:
                        pck_brightness_map[pck_score] = []
                    pck_brightness_map[pck_score].append(brightness)

            # Calculate distributions for each PCK score
            threshold_distributions = {}

            for pck_score, brightness_list in pck_brightness_map.items():
                if len(brightness_list) < 2:  # Need at least 2 values for distribution
                    continue

                # Create brightness bins
                min_brightness = min(brightness_list)
                max_brightness = max(brightness_list)

                # Ensure we have a reasonable range
                if max_brightness - min_brightness < bin_size:
                    bin_edges = np.linspace(
                        min_brightness - bin_size / 2,
                        max_brightness + bin_size / 2,
                        max(
                            3,
                            int(
                                (max_brightness - min_brightness + bin_size) / bin_size
                            ),
                        ),
                    )
                else:
                    bin_edges = np.arange(
                        min_brightness, max_brightness + bin_size, bin_size
                    )

                # Calculate histogram
                frequencies, _ = np.histogram(brightness_list, bins=bin_edges)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

                # Normalize frequencies
                total_frames = len(brightness_list)
                normalized_frequencies = (
                    frequencies / total_frames if total_frames > 0 else frequencies
                )

                distribution = {
                    "pck_score": pck_score,
                    "total_frames": total_frames,
                    "bin_centers": bin_centers.tolist(),
                    "frequencies": frequencies.tolist(),
                    "normalized_frequencies": normalized_frequencies.tolist(),
                    "brightness_range": [min_brightness, max_brightness],
                    "mean_brightness": np.mean(brightness_list),
                    "std_brightness": np.std(brightness_list),
                }

                threshold_distributions[pck_score] = distribution

            distributions[pck_threshold] = threshold_distributions
            print(
                f"     Calculated distributions for {len(threshold_distributions)} PCK scores"
            )

        return distributions

    def calculate_joint_brightness_distributions(
        self, analysis_results: Dict, bin_size: int = 5
    ) -> Dict:
        """Calculate brightness distributions for each joint."""
        if "brightness_data" not in analysis_results:
            print("❌ No brightness data in analysis results")
            return {}

        print(f"Calculating joint brightness distributions with bin size: {bin_size}")

        joint_distributions = {}

        for joint_name, brightness_values in analysis_results[
            "brightness_data"
        ].items():
            print(f"   Processing joint: {joint_name}")

            # Filter out NaN values
            valid_brightness = [b for b in brightness_values if not np.isnan(b)]

            if len(valid_brightness) < 2:
                print(f"     ⚠️  Insufficient data for {joint_name}")
                continue

            # Create brightness bins
            min_brightness = min(valid_brightness)
            max_brightness = max(valid_brightness)

            if max_brightness - min_brightness < bin_size:
                bin_edges = np.linspace(
                    min_brightness - bin_size / 2,
                    max_brightness + bin_size / 2,
                    max(
                        3, int((max_brightness - min_brightness + bin_size) / bin_size)
                    ),
                )
            else:
                bin_edges = np.arange(
                    min_brightness, max_brightness + bin_size, bin_size
                )

            # Calculate histogram
            frequencies, _ = np.histogram(valid_brightness, bins=bin_edges)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            # Normalize frequencies
            total_frames = len(valid_brightness)
            normalized_frequencies = (
                frequencies / total_frames if total_frames > 0 else frequencies
            )

            distribution = {
                "joint_name": joint_name,
                "total_frames": total_frames,
                "bin_centers": bin_centers.tolist(),
                "frequencies": frequencies.tolist(),
                "normalized_frequencies": normalized_frequencies.tolist(),
                "brightness_range": [min_brightness, max_brightness],
                "mean_brightness": np.mean(valid_brightness),
                "std_brightness": np.std(valid_brightness),
            }

            joint_distributions[joint_name] = distribution
            print(f"     ✅ Distribution calculated: {total_frames} frames")

        return joint_distributions

    def calculate_pck_score_distributions(
        self, analysis_results: Dict, joint_name: str = None
    ) -> Dict:
        """Calculate PCK score distributions."""
        if "pck_data" not in analysis_results:
            print("❌ No PCK data in analysis results")
            return {}

        print("Calculating PCK score distributions")

        # Use first joint if none specified
        if joint_name is None:
            joint_name = list(analysis_results["pck_data"].keys())[0]
            print(f"Using joint: {joint_name}")

        if joint_name not in analysis_results["pck_data"]:
            print(f"❌ Joint {joint_name} not found in analysis results")
            return {}

        joint_data = analysis_results["pck_data"][joint_name]
        pck_distributions = {}

        for pck_threshold, pck_scores in joint_data["pck_scores"].items():
            print(f"   Processing PCK threshold: {pck_threshold}")

            # Calculate PCK score distribution
            unique_scores, counts = np.unique(pck_scores, return_counts=True)
            total_frames = len(pck_scores)

            # Normalize frequencies
            normalized_frequencies = (
                counts / total_frames if total_frames > 0 else counts
            )

            distribution = {
                "pck_threshold": pck_threshold,
                "total_frames": total_frames,
                "unique_scores": unique_scores.tolist(),
                "frequencies": counts.tolist(),
                "normalized_frequencies": normalized_frequencies.tolist(),
                "mean_pck": np.mean(pck_scores),
                "std_pck": np.std(pck_scores),
                "score_range": [int(min(pck_scores)), int(max(pck_scores))],
            }

            pck_distributions[pck_threshold] = distribution
            print(
                f"     ✅ Distribution calculated: {len(unique_scores)} unique scores"
            )

        return pck_distributions

    def generate_distribution_summary(
        self, brightness_distributions: Dict, pck_distributions: Dict
    ) -> pd.DataFrame:
        """Generate summary of all distributions."""
        summary_data = []

        # Brightness distribution summaries
        for joint_name, dist in brightness_distributions.items():
            summary_data.append(
                {
                    "type": "brightness",
                    "identifier": joint_name,
                    "total_frames": dist["total_frames"],
                    "mean_value": dist["mean_brightness"],
                    "std_value": dist["std_brightness"],
                    "value_range": f"{dist['brightness_range'][0]:.1f}-{dist['brightness_range'][1]:.1f}",
                    "num_bins": len(dist["bin_centers"]),
                    "peak_frequency": max(dist["normalized_frequencies"]),
                }
            )

        # PCK distribution summaries
        for pck_threshold, dist in pck_distributions.items():
            summary_data.append(
                {
                    "type": "pck_scores",
                    "identifier": pck_threshold,
                    "total_frames": dist["total_frames"],
                    "mean_value": dist["mean_pck"],
                    "std_value": dist["std_pck"],
                    "value_range": f"{dist['score_range'][0]}-{dist['score_range'][1]}",
                    "num_bins": len(dist["unique_scores"]),
                    "peak_frequency": max(dist["normalized_frequencies"]),
                }
            )

        summary_df = pd.DataFrame(summary_data)

        if not summary_df.empty:
            # Round numeric columns
            numeric_columns = summary_df.select_dtypes(include=[np.number]).columns
            summary_df[numeric_columns] = summary_df[numeric_columns].round(4)

        return summary_df

    def export_distributions(
        self,
        brightness_distributions: Dict,
        pck_distributions: Dict,
        pck_brightness_distributions: Dict,
        output_filename: str = None,
    ) -> str:
        """Export all distributions to files."""
        try:
            if output_filename is None:
                output_filename = f"gt_distributions_{self.dataset_name}"

            from config import ConfigManager

            config = ConfigManager.load_config(self.dataset_name)
            output_dir = config.save_folder
            os.makedirs(output_dir, exist_ok=True)

            # Export as JSON
            import json

            export_data = {
                "dataset": self.dataset_name,
                "brightness_distributions": brightness_distributions,
                "pck_distributions": pck_distributions,
                "pck_brightness_distributions": pck_brightness_distributions,
                "metadata": {
                    "generated_by": "GTDistributionCalculator",
                    "total_joints": len(brightness_distributions),
                    "total_pck_thresholds": len(pck_distributions),
                },
            }

            json_path = os.path.join(output_dir, f"{output_filename}.json")
            with open(json_path, "w") as f:
                json.dump(export_data, f, indent=2, default=str)

            print(f"✅ Exported distributions to: {json_path}")

            # Export summary CSV
            summary_df = self.generate_distribution_summary(
                brightness_distributions, pck_distributions
            )

            if not summary_df.empty:
                csv_path = os.path.join(output_dir, f"{output_filename}_summary.csv")
                summary_df.to_csv(csv_path, index=False)
                print(f"✅ Exported summary to: {csv_path}")

            return json_path

        except Exception as e:
            print(f"❌ Failed to export distributions: {e}")
            return ""

    def run_complete_distribution_analysis(
        self,
        joint_names: List[str] = None,
        pck_threshold: str = None,
        bin_size: int = 5,
        video_path: str = None,
        **gt_filter_kwargs,
    ) -> Dict:
        """Run complete distribution analysis."""
        print(f"🔍 Starting complete distribution analysis for {self.dataset_name}")

        try:
            # 1. Run PCK-brightness analysis
            print("\n📊 Step 1: Running PCK-brightness analysis...")
            analysis_results = self.analyzer.analyze_pck_brightness_correlation(
                joint_names=joint_names,
                pck_threshold=pck_threshold,
                video_path=video_path,
                **gt_filter_kwargs,
            )

            if not analysis_results.get("brightness_data"):
                print("❌ No analysis results available")
                return {}

            # 2. Calculate brightness distributions by PCK score
            print("\n📈 Step 2: Calculating brightness distributions by PCK score...")
            pck_brightness_distributions = (
                self.calculate_brightness_distributions_by_pck(
                    analysis_results, bin_size
                )
            )

            # 3. Calculate joint brightness distributions
            print("\n🔆 Step 3: Calculating joint brightness distributions...")
            joint_brightness_distributions = (
                self.calculate_joint_brightness_distributions(
                    analysis_results, bin_size
                )
            )

            # 4. Calculate PCK score distributions
            print("\n📋 Step 4: Calculating PCK score distributions...")
            pck_score_distributions = self.calculate_pck_score_distributions(
                analysis_results
            )

            # Compile results
            distribution_results = {
                "dataset": self.dataset_name,
                "analysis_results": analysis_results,
                "pck_brightness_distributions": pck_brightness_distributions,
                "joint_brightness_distributions": joint_brightness_distributions,
                "pck_score_distributions": pck_score_distributions,
                "parameters": {
                    "bin_size": bin_size,
                    "joint_names": joint_names,
                    "pck_threshold": pck_threshold,
                },
            }

            print("✅ Complete distribution analysis finished")
            return distribution_results

        except Exception as e:
            print(f"❌ Distribution analysis failed: {e}")
            import traceback

            traceback.print_exc()
            return {}


def main():
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Ground Truth Distribution Calculator")
    parser.add_argument("dataset", help="Dataset name (e.g., 'humaneva', 'movi')")
    parser.add_argument(
        "--joints", nargs="*", help="Specific joints to analyze (default: all)"
    )
    parser.add_argument("--pck-threshold", help="Specific PCK threshold to analyze")
    parser.add_argument("--bin-size", type=int, default=5, help="Brightness bin size")
    parser.add_argument("--video", help="Path to specific video file")
    parser.add_argument("--subject", help="Filter ground truth by subject")
    parser.add_argument("--action", help="Filter ground truth by action")
    parser.add_argument("--camera", type=int, help="Filter ground truth by camera")
    parser.add_argument("--export", action="store_true", help="Export distributions")
    parser.add_argument("--filename", help="Custom output filename")
    parser.add_argument("--summary-only", action="store_true", help="Show summary only")

    args = parser.parse_args()

    try:
        calculator = GTDistributionCalculator(args.dataset)

        # Set up ground truth filters
        gt_filters = {}
        if args.subject:
            gt_filters["subject"] = args.subject
        if args.action:
            gt_filters["action"] = args.action
        if args.camera is not None:
            gt_filters["camera"] = args.camera

        # Run complete analysis
        results = calculator.run_complete_distribution_analysis(
            joint_names=args.joints,
            pck_threshold=args.pck_threshold,
            bin_size=args.bin_size,
            video_path=args.video,
            **gt_filters,
        )

        if not results:
            print("❌ No distribution results available")
            return

        # Display summary
        summary_df = calculator.generate_distribution_summary(
            results["joint_brightness_distributions"],
            results["pck_score_distributions"],
        )

        if not summary_df.empty:
            print("\n" + "=" * 80)
            print("DISTRIBUTION SUMMARY")
            print("=" * 80)
            print(summary_df.to_string(index=False))

        # Show detailed results if not summary-only
        if not args.summary_only:
            print("\n" + "=" * 80)
            print("DETAILED DISTRIBUTION ANALYSIS")
            print("=" * 80)

            # Show PCK-brightness distributions
            for threshold, pck_dists in results["pck_brightness_distributions"].items():
                print(f"\nPCK Threshold: {threshold}")
                for pck_score, dist in pck_dists.items():
                    print(
                        f"  PCK {pck_score}: {dist['total_frames']} frames, "
                        f"mean brightness: {dist['mean_brightness']:.2f}"
                    )

        # Export results
        if args.export:
            calculator.export_distributions(
                results["joint_brightness_distributions"],
                results["pck_score_distributions"],
                results["pck_brightness_distributions"],
                args.filename,
            )

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

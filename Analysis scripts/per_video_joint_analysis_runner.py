"""
Per-Video Joint Analysis Runner

Runner for per-video joint brightness analysis that analyzes all joints for each video.
"""

from utils.config_extractor import extract_joint_analysis_config
from core.joint_data_loader import JointDataLoader
from visualizers.per_video_joint_brightness_visualizer import (
    PerVideoJointBrightnessVisualizer,
)
from analyzers.per_video_joint_brightness_analyzer import (
    PerVideoJointBrightnessAnalyzer,
)
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

# Import the per-video analyzer and visualizer

# Import existing data loader and config extractor


class PerVideoJointAnalysisRunner:
    """Runner for per-video joint brightness analysis."""

    def __init__(
        self,
        dataset_name: str = "movi",
        joint_names: list = None,
        output_dir: str = None,
        save_results: bool = True,
        sampling_radius: int = 3,
    ):
        """Initialize the per-video analysis runner.

        Args:
            dataset_name: Name of the dataset to analyze ('movi' or 'humaneva')
            joint_names: List of joint names to analyze (uses default if None)
            output_dir: Output directory (auto-generated if None)
            save_results: Whether to save results to files
            sampling_radius: Radius for brightness sampling around joints
        """
        self.dataset_name = dataset_name

        # Get joints from config or use provided/default joints
        config_joints, config_pck_thresholds = extract_joint_analysis_config(
            dataset_name
        )
        self.joint_names = joint_names or config_joints or self._get_default_joints()
        self.pck_thresholds = config_pck_thresholds or [0.01, 0.02, 0.05]

        self.save_results = save_results
        self.sampling_radius = sampling_radius

        # Setup output directory
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = Path(
                f"/storage/Projects/Gaitly/bsehgal/lower_body_pose_est/pipeline_results/analysis_results/"
                f"{dataset_name.upper()}/per_video_joint_analysis/per_video_analysis_{timestamp}"
            )

        if self.save_results:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        print("Per-Video Joint Analysis Runner")
        print(f"Dataset: {self.dataset_name}")
        print(f"Joints: {', '.join(self.joint_names)}")
        print(f"PCK Thresholds: {self.pck_thresholds}")
        if self.save_results:
            print(f"Output: {self.output_dir}")

    def _get_default_joints(self) -> list:
        """Get default lower body joints (matching MoVi config format)."""
        return [
            "LEFT_HIP",
            "RIGHT_HIP",
            "LEFT_KNEE",
            "RIGHT_KNEE",
            "LEFT_ANKLE",
            "RIGHT_ANKLE",
        ]

    def run_analysis(self) -> Dict[str, Any]:
        """Run the complete per-video joint analysis.

        Returns:
            Dict containing analysis results for each video
        """
        print("\\n" + "=" * 60)
        print("STARTING PER-VIDEO JOINT BRIGHTNESS ANALYSIS")
        print("=" * 60)

        try:
            # Step 1: Load data
            print("\\n1. Loading and validating data...")
            data_loader = JointDataLoader(self.dataset_name)

            # Setup configuration and load PCK data
            if not data_loader.setup_configuration():
                print("Failed to load configuration")
                return {}

            # Load PCK data - this will load all available PCK columns from the Excel file
            # including all jointwise PCK columns with their thresholds (0.01, 0.02, 0.05)
            pck_data = data_loader.load_pck_data()

            if pck_data is None:
                print("Failed to load data")
                return {}

            print(f"Loaded data with {len(pck_data)} rows")
            print(f"   Columns: {list(pck_data.columns)}")

            # Check if we have video grouping columns
            if "video_file" in pck_data.columns:
                video_count = pck_data["video_file"].nunique()
                print(f"   Found {video_count} unique videos")
            elif "subject" in pck_data.columns:
                video_count = pck_data["subject"].nunique()
                print(f"   Found {video_count} unique subjects (as videos)")
            else:
                print("❌ No video grouping column found")
                return {}

            # Step 2: Initialize analyzer
            print("\n 2. Initializing per-video analyzer...")

            # Use the loaded configuration from data_loader
            config = data_loader.config

            analyzer = PerVideoJointBrightnessAnalyzer(
                config=config,
                joint_names=self.joint_names,
                sampling_radius=self.sampling_radius,
                dataset_name=self.dataset_name,
            )

            # Step 3: Run analysis
            print("\\n3. Running per-video analysis...")
            analysis_results = analyzer.analyze(pck_data)

            if not analysis_results:
                print("❌ No analysis results generated")
                return {}

            print(f"✅ Analysis completed for {len(analysis_results)} videos")

            # Step 4: Create visualizations
            if self.save_results:
                print("\\n4. Creating visualizations...")
                visualizer = PerVideoJointBrightnessVisualizer(
                    output_dir=str(self.output_dir),
                    save_plots=True,
                    create_individual_plots=False,  # Focus on combined plot
                )

                visualizer.create_all_visualizations(analysis_results)
                visualizer.save_results_to_csv(analysis_results)
                print("✅ Visualizations and CSV export completed")

            # Step 5: Print summary
            print("\\n5. Analysis Summary:")
            self._print_summary(analysis_results)

            print("\\n" + "=" * 60)
            print("PER-VIDEO JOINT ANALYSIS COMPLETED SUCCESSFULLY!")
            if self.save_results:
                print(f"Results saved to: {self.output_dir}")
            print("=" * 60)

            return analysis_results

        except Exception as e:
            print(f"\\n❌ Analysis failed: {e}")
            import traceback

            traceback.print_exc()
            return {}

    def _print_summary(self, analysis_results: Dict[str, Any]) -> None:
        """Print a summary of the analysis results."""

        print(f"   📊 Total videos analyzed: {len(analysis_results)}")

        # Count total joints and PCK metrics analyzed
        total_joints = set()
        total_pck_metrics = 0
        total_frames = 0

        correlations = []

        for video_name, video_results in analysis_results.items():
            joints_analyzed = video_results.get("joints_analyzed", [])
            total_joints.update(joints_analyzed)
            total_frames += video_results.get("total_frames", 0)

            # Count PCK metrics and collect correlations
            for pck_column, pck_results in video_results.items():
                if pck_column in [
                    "video_name",
                    "total_frames",
                    "joints_analyzed",
                    "brightness_summary",
                ]:
                    continue

                total_pck_metrics += 1

                if "correlation" in pck_results:
                    corr = pck_results["correlation"]["pearson"]
                    if not pd.isna(corr):
                        correlations.append(corr)

        print(
            f"   🦴 Unique joints analyzed: {len(total_joints)} ({', '.join(sorted(total_joints))})"
        )
        print(f"   📈 Total PCK metrics: {total_pck_metrics}")
        print(f"   🎬 Total frames processed: {total_frames}")

        if correlations:
            print("   📊 Correlation statistics:")
            print(f"      - Mean correlation: {np.mean(correlations):.3f}")
            print(f"      - Std correlation: {np.std(correlations):.3f}")
            print(
                f"      - Strong correlations (|r| > 0.5): {sum(1 for c in correlations if abs(c) > 0.5)}"
            )
            print(
                f"      - Positive correlations: {sum(1 for c in correlations if c > 0.1)}"
            )
            print(
                f"      - Negative correlations: {sum(1 for c in correlations if c < -0.1)}"
            )

        # Find videos with highest/lowest average brightness
        brightness_averages = {}
        for video_name, video_results in analysis_results.items():
            brightness_summary = video_results.get("brightness_summary", {})
            if brightness_summary:
                avg_brightness = np.mean(
                    [
                        stats["mean"]
                        for stats in brightness_summary.values()
                        if stats["mean"] > 0
                    ]
                )
                brightness_averages[video_name] = avg_brightness

        if brightness_averages:
            brightest_video = max(brightness_averages,
                                  key=brightness_averages.get)
            darkest_video = min(brightness_averages,
                                key=brightness_averages.get)
            print(
                f"   💡 Brightest video: {brightest_video} ({brightness_averages[brightest_video]:.2f})"
            )
            print(
                f"   🌑 Darkest video: {darkest_video} ({brightness_averages[darkest_video]:.2f})"
            )


def run_per_video_analysis(
    dataset_name: str = "movi",
    joint_names: list = None,
    output_dir: str = None,
    save_results: bool = True,
    sampling_radius: int = 3,
) -> Dict[str, Any]:
    """Run per-video joint brightness analysis.

    Args:
        dataset_name: Name of dataset ('movi' or 'humaneva')
        joint_names: List of joint names (uses default if None)
        output_dir: Output directory (auto-generated if None)
        save_results: Whether to save results and visualizations
        sampling_radius: Radius for brightness sampling

    Returns:
        Dict containing analysis results for each video
    """
    runner = PerVideoJointAnalysisRunner(
        dataset_name=dataset_name,
        joint_names=joint_names,
        output_dir=output_dir,
        save_results=save_results,
        sampling_radius=sampling_radius,
    )

    return runner.run_analysis()


if __name__ == "__main__":
    """Example usage."""
    import pandas as pd

    # Run analysis for MoVi dataset
    results = run_per_video_analysis(
        dataset_name="movi",
        joint_names=None,  # Use default joints
        save_results=True,
        sampling_radius=3,
    )

    if results:
        print(f"\n🎉 Successfully analyzed {len(results)} videos!")
        print("\nExample video results:")
        for i, (video_name, video_data) in enumerate(results.items()):
            if i >= 3:  # Show only first 3 videos
                break
            print(
                f"  - {video_name}: {len(video_data.get('joints_analyzed', []))} joints, "
                f"{video_data.get('total_frames', 0)} frames"
            )

        # Visualize first frame with joint circles for first video
        print("\nVisualizing first frame with joint circles for first video...")
        # Re-initialize analyzer to access method
        from analyzers.per_video_joint_brightness_analyzer import (
            PerVideoJointBrightnessAnalyzer,
        )
        from core.joint_data_loader import JointDataLoader

        data_loader = JointDataLoader("movi")
        data_loader.setup_configuration()
        pck_data = data_loader.load_pck_data()
        config = data_loader.config
        analyzer = PerVideoJointBrightnessAnalyzer(
            config=config,
            joint_names=None,
            sampling_radius=3,
            dataset_name="movi",
        )
        # Get grouping columns and group key for first video
        grouping_cols = config.get_grouping_columns()
        first_video_name = list(results.keys())[0]
        # Find group_key for first video
        group_key = None
        for key, group in pck_data.groupby(grouping_cols):
            video_name = config.create_video_name(key, grouping_cols)
            if video_name == first_video_name:
                group_key = key
                video_data = group
                break
        if group_key is not None:
            analyzer.visualize_first_frame_with_joint_circles(
                first_video_name,
                video_data,
                group_key=group_key,
                grouping_cols=grouping_cols,
                output_dir=None,  # Set to a path to save instead of show
            )
        else:
            print("Could not find group key for first video.")
    else:
        print("\n❌ Analysis failed or returned no results")

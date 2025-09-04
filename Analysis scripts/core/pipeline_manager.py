"""
Pipeline Manager Module

Handles the main analysis pipeline coordination and execution.
"""

from config import ConfigManager
from core.data_processor import DataProcessor
from core.visualization_manager import VisualizationManager
from core.statistical_analysis_manager import StatisticalAnalysisManager
from utils import PerformanceMonitor, ProgressTracker
import os
from datetime import datetime


class AnalysisPipeline:
    """Main analysis pipeline coordinator."""

    def __init__(self, dataset_name: str):
        """Initialize the analysis pipeline."""
        self.dataset_name = dataset_name
        self.config = ConfigManager.load_config(dataset_name)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(self.config.save_folder, exist_ok=True)

        # Initialize managers
        self.data_processor = DataProcessor(self.config)
        self.viz_manager = VisualizationManager(self.config, self.timestamp)
        self.stats_manager = StatisticalAnalysisManager(self.config, self.timestamp)

    @PerformanceMonitor.timing_decorator
    def run_complete_analysis(
        self,
        metrics_config: dict,
        run_overall: bool,
        run_per_video: bool,
        run_per_frame: bool,
        per_frame_analysis_types: list,
    ):
        """Run complete analysis pipeline."""
        print(f"Starting analysis pipeline for {self.dataset_name.upper()} dataset...")

        if run_overall:
            self._run_overall_analysis(metrics_config)
            print("\n" + "=" * 50 + "\n")

        if run_per_video:
            self._run_per_video_analysis(metrics_config)
            print("\n" + "=" * 50 + "\n")

        if run_per_frame:
            self._run_per_frame_analysis(metrics_config, per_frame_analysis_types)

        print(
            f"\nComplete analysis pipeline finished for {self.dataset_name.upper()} dataset."
        )

    def _run_overall_analysis(self, metrics_config: dict):
        """Run overall analysis using separated components."""
        print("Running overall analysis with separated components...")

        pck_df = self.data_processor.load_pck_scores()
        if pck_df is None:
            print("Cannot proceed with overall analysis without data.")
            return

        results = self.data_processor.process_overall_data(pck_df, metrics_config)

        # Create visualizations for each metric
        for metric_name, metric_data in results.items():
            self.viz_manager.create_overall_visualizations(
                metric_data["merged_df"], metric_data["all_metric_data"], metric_name
            )

        print(f"Overall analysis complete. Results saved to {self.config.save_folder}")

    def _run_per_video_analysis(self, metrics_config: dict):
        """Run per-video analysis with video-level aggregation."""
        print("Running per-video analysis with video-level aggregation...")

        # Load overall PCK data for per-video analysis
        pck_df = self.data_processor.load_pck_scores()
        if pck_df is None:
            print("Cannot proceed with per-video analysis without PCK data.")
            return

        # Process the data for per-video analysis
        per_video_results = self.data_processor.process_overall_data(
            pck_df, metrics_config
        )

        # Create per-video visualizations
        for metric_name, metric_data in per_video_results.items():
            self.viz_manager.create_per_video_visualizations(
                metric_data["merged_df"], metric_name
            )

        # Create PCK vs brightness correlation plot using processed data with brightness
        # Use the first available metric_data's merged_df which should contain brightness after processing
        if per_video_results:
            first_metric = next(iter(per_video_results.values()))
            merged_df = first_metric["merged_df"]

            # Get the grouping columns from config to determine the video identifier column
            grouping_columns = self.config.get_grouping_columns()

            if (
                merged_df is not None
                and not merged_df.empty
                and "avg_brightness" in merged_df.columns
                and grouping_columns
                and any(col in merged_df.columns for col in grouping_columns)
            ):
                # Use the first available grouping column as the video identifier
                video_id_col = next(
                    col for col in grouping_columns if col in merged_df.columns
                )

                print(
                    f"Creating PCK vs brightness correlation plot for per-video analysis using '{video_id_col}' as identifier..."
                )
                self.viz_manager.create_pck_brightness_correlation_plot(
                    merged_df,
                    brightness_col="avg_brightness",
                    video_id_col=video_id_col,
                    analysis_type="per_video",
                )
            else:
                missing_items = []
                if "avg_brightness" not in merged_df.columns:
                    missing_items.append("avg_brightness column")
                if not grouping_columns:
                    missing_items.append("grouping columns in config")
                elif not any(col in merged_df.columns for col in grouping_columns):
                    missing_items.append(
                        f"any of the configured grouping columns {grouping_columns}"
                    )

                print(
                    f"Skipping PCK vs brightness correlation plot - missing: {', '.join(missing_items)}"
                )
        else:
            print("No processed data available for correlation plot")

        print(
            f"Per-video analysis complete. Results saved to {self.config.save_folder}"
        )

    def _run_per_frame_analysis(self, metrics_config: dict, analysis_types: list):
        """Run per-frame analysis using separated components."""
        print("Running per-frame analysis with separated components...")

        pck_df = self.data_processor.load_pck_per_frame_scores()
        if pck_df is None:
            print("Cannot proceed with per-frame analysis without data.")
            return

        combined_df = self.data_processor.process_per_frame_data(pck_df, metrics_config)

        if combined_df.empty:
            print("No combined data to analyze.")
            return

        # Run statistical analyses
        self.stats_manager.run_statistical_analyses(
            combined_df, metrics_config, analysis_types
        )

        # Run PCK brightness analysis with scenarios if available and multi-analysis is enabled
        if "pck_brightness" in analysis_types and self.config.analysis_config:
            multi_analysis_config = self.config.analysis_config.get(
                "pck_brightness", {}
            ).get("multi_analysis", {})

            # Only run scenarios in single analysis mode (not when called from multi-analysis pipeline)
            if multi_analysis_config.get("enabled", False) and not hasattr(
                self, "_in_multi_analysis_context"
            ):
                scenarios = multi_analysis_config.get("scenarios", [])
                enabled_scenarios = [s for s in scenarios if s.get("enabled", False)]

                if enabled_scenarios:
                    print(
                        f"\nRunning PCK brightness analysis with {len(enabled_scenarios)} scenarios..."
                    )

                    # Get score groups config for resolving score group names
                    pck_brightness_config = self.config.analysis_config.get(
                        "pck_brightness", {}
                    )
                    score_groups_config = pck_brightness_config.get("score_groups", {})

                    for scenario in enabled_scenarios:
                        scenario_name = scenario.get("name", "unnamed_scenario")
                        score_group_name = scenario.get("score_group", "all")
                        description = scenario.get("description", "No description")

                        # Get the actual score group values
                        score_group = score_groups_config.get(score_group_name)

                        print(f"\nProcessing scenario '{scenario_name}':")
                        print(f"  Description: {description}")
                        print(f"  Score group: {score_group_name} -> {score_group}")

                        if (
                            score_group is not None
                        ):  # Skip groups with null values (like 'all')
                            success = self.stats_manager.run_pck_brightness_analysis_with_score_groups(
                                combined_df, score_group, scenario_name
                            )
                            if success:
                                print(f"  ✅ Scenario '{scenario_name}' completed")
                            else:
                                print(f"  ❌ Scenario '{scenario_name}' failed")
                        else:
                            print(
                                f"  ⚠️  Skipping scenario '{scenario_name}' (null score group)"
                            )

        # Create visualizations for each metric
        for metric_name in metrics_config.keys():
            self.viz_manager.create_per_frame_visualizations(combined_df, metric_name)

        # Create PCK vs brightness correlation plot if brightness data is available
        if "brightness" in combined_df.columns and "video_id" in combined_df.columns:
            print(
                "Creating PCK vs brightness correlation plot for per-frame analysis..."
            )
            self.viz_manager.create_pck_brightness_correlation_plot(
                combined_df, analysis_type="per_frame"
            )

        print(
            f"Per-frame analysis complete. Results saved to {self.config.save_folder}"
        )

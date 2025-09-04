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
        run_per_frame: bool,
        per_frame_analysis_types: list,
    ):
        """Run complete analysis pipeline."""
        print(f"Starting analysis pipeline for {self.dataset_name.upper()} dataset...")

        if run_overall:
            self._run_overall_analysis(metrics_config)
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

        # Create visualizations for each metric
        for metric_name in metrics_config.keys():
            self.viz_manager.create_per_frame_visualizations(combined_df, metric_name)

        # Create PCK vs brightness correlation plot if brightness data is available
        if "brightness" in combined_df.columns and "video_id" in combined_df.columns:
            self.viz_manager.create_pck_brightness_correlation_plot(combined_df)

        print(
            f"Per-frame analysis complete. Results saved to {self.config.save_folder}"
        )

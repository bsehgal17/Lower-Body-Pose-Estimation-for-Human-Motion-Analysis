"""
Modularized Analysis Pipeline

Simplified main entry point for analysis workflows.
"""

import os
from datetime import datetime
from analyzers import AnalyzerFactory
from visualizers import VisualizationFactory
from utils import PerformanceMonitor

from core.config_manager import AnalysisConfigManager
from core.data_processor import DataProcessor
from core.visualization_manager import VisualizationManager
from core.statistical_analysis_manager import StatisticalAnalysisManager
from core.multi_analysis_pipeline import MultiAnalysisPipeline


class AnalysisPipeline:
    """Simplified analysis pipeline using modularized components."""

    def __init__(self, dataset_name: str):
        """Initialize the analysis pipeline."""
        self.dataset_name = dataset_name
        self.config = AnalysisConfigManager.load_dataset_config(dataset_name)
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

        success = True

        if run_overall:
            success &= self._run_overall_analysis(metrics_config)
            print("\n" + "=" * 50 + "\n")

        if run_per_frame:
            success &= self._run_per_frame_analysis(
                metrics_config, per_frame_analysis_types
            )

        print(
            f"\nComplete analysis pipeline finished for {self.dataset_name.upper()} dataset."
        )
        return success

    def _run_overall_analysis(self, metrics_config: dict):
        """Run overall analysis using modularized components."""
        print("Running overall analysis with separated components...")

        pck_df = self.data_processor.load_pck_scores()
        if pck_df is None:
            print("Cannot proceed with overall analysis without data.")
            return False

        results = self.data_processor.process_overall_data(pck_df, metrics_config)

        for metric_name, metric_data in results.items():
            self.viz_manager.create_overall_visualizations(
                metric_data["merged_df"], metric_data["all_metric_data"], metric_name
            )

        # Create PCK line plot for overall analysis if data is available
        if results:
            first_result = next(iter(results.values()))
            if "merged_df" in first_result and first_result["merged_df"] is not None:
                self.viz_manager.create_pck_line_plot(
                    first_result["merged_df"], analysis_type="overall"
                )
                self.viz_manager.create_individual_pck_plots(
                    first_result["merged_df"], analysis_type="overall"
                )

        print(f"Overall analysis complete. Results saved to {self.config.save_folder}")
        return True

    def _run_per_frame_analysis(self, metrics_config: dict, analysis_types: list):
        """Run per-frame analysis using modularized components."""
        print("Running per-frame analysis with separated components...")

        pck_df = self.data_processor.load_pck_per_frame_scores()
        if pck_df is None:
            print("Cannot proceed with per-frame analysis without data.")
            return False

        combined_df = self.data_processor.process_per_frame_data(pck_df, metrics_config)

        if combined_df.empty:
            print("No combined data to analyze.")
            return False

        # Run analyses with progress tracking
        from utils import ProgressTracker

        progress = ProgressTracker(
            len(metrics_config) * len(analysis_types), "Running statistical analyses"
        )

        for metric_name in metrics_config.keys():
            self.stats_manager.run_statistical_analyses(
                combined_df, metric_name, analysis_types, progress
            )
            self.viz_manager.create_per_frame_visualizations(combined_df, metric_name)

        progress.finish()

        # Create PCK line plots
        self.viz_manager.create_pck_line_plot(combined_df, analysis_type="per_frame")
        self.viz_manager.create_individual_pck_plots(
            combined_df, analysis_type="per_frame"
        )

        print(
            f"Per-frame analysis complete. Results saved to {self.config.save_folder}"
        )
        return True


def run_single_analysis(dataset_name: str):
    """Run single analysis mode."""
    print("🚀 Running Single Analysis Pipeline")
    print("=" * 70)

    metrics_config = AnalysisConfigManager.get_metrics_config()

    pipeline = AnalysisPipeline(dataset_name)
    success = pipeline.run_complete_analysis(
        metrics_config=metrics_config,
        run_overall=True,
        run_per_frame=True,
        per_frame_analysis_types=AnalyzerFactory.get_available_analyzers(),
    )

    if success:
        print("✅ Single analysis completed successfully")
    else:
        print("❌ Single analysis completed with errors")

    return success


def run_multi_analysis(dataset_name: str):
    """Run multi-analysis mode."""
    analysis_config = AnalysisConfigManager.load_analysis_config()
    dataset_config = AnalysisConfigManager.load_dataset_config(dataset_name)
    metrics_config = AnalysisConfigManager.get_metrics_config()

    # Create a pipeline instance for multi-analysis
    pipeline = AnalysisPipeline(dataset_name)
    multi_pipeline = MultiAnalysisPipeline(
        dataset_config, pipeline.data_processor, pipeline.timestamp
    )

    results = multi_pipeline.run_multi_analysis(
        analysis_config, dataset_config, metrics_config
    )

    return results


def test_components():
    """Test analysis components."""
    print("\n" + "=" * 60)
    print("Testing analysis components...")

    analysis_config = AnalysisConfigManager.test_configurations()

    print(f"Available analyzers: {AnalyzerFactory.get_available_analyzers()}")
    print(f"Available visualizers: {VisualizationFactory.get_available_visualizers()}")
    print(f"Per-frame analysis types: {AnalyzerFactory.get_available_analyzers()}")
    print(
        f"Global analysis score groups: {analysis_config.get_available_score_groups()}"
    )
    print("Components test complete.\n")

    return analysis_config


def main():
    """Main entry point for analysis."""
    dataset_name = "movi"

    try:
        # Test components first
        analysis_config = test_components()

        # Check if multi-analysis is enabled in config
        if analysis_config.is_multi_analysis_enabled():
            results = run_multi_analysis(dataset_name)
        else:
            success = run_single_analysis(dataset_name)

    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

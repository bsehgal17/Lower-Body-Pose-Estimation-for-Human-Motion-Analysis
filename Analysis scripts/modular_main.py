"""
Modular analysis pipeline using separated components.
"""

from core.config import ConfigManager
from core.analyzers import AnalyzerFactory
from core.visualizers import VisualizationFactory
from unified_data_processor import UnifiedDataProcessor
from utils import PerformanceMonitor, ProgressTracker
import os
from datetime import datetime


class ModularAnalysisPipeline:
    """Modular analysis pipeline using separated components."""

    def __init__(self, dataset_name: str):
        """Initialize the analysis pipeline."""
        self.dataset_name = dataset_name
        self.config = ConfigManager.load_config(dataset_name)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(self.config.save_folder, exist_ok=True)
        self.data_processor = UnifiedDataProcessor(self.config)

    @PerformanceMonitor.timing_decorator
    def run_complete_analysis(
        self,
        metrics_config: dict,
        run_overall: bool,
        run_per_frame: bool,
        per_frame_analysis_types: list,
    ):
        """Run complete analysis pipeline."""
        print(
            f"Starting modular analysis pipeline for {self.dataset_name.upper()} dataset..."
        )

        if run_overall:
            self._run_overall_analysis(metrics_config)
            print("\n" + "=" * 50 + "\n")

        if run_per_frame:
            self._run_per_frame_analysis(metrics_config, per_frame_analysis_types)

        print(
            f"\nComplete modular analysis pipeline finished for {self.dataset_name.upper()} dataset."
        )

    def _run_overall_analysis(self, metrics_config: dict):
        """Run overall analysis using modular components."""
        print("Running overall analysis with modular components...")

        pck_df = self.data_processor.load_pck_scores()
        if pck_df is None:
            print("Cannot proceed with overall analysis without data.")
            return

        results = self.data_processor.process_overall_data(pck_df, metrics_config)

        for metric_name, metric_data in results.items():
            self._create_overall_visualizations(
                metric_data["merged_df"], metric_data["all_metric_data"], metric_name
            )

        print(f"Overall analysis complete. Results saved to {self.config.save_folder}")

    def _run_per_frame_analysis(self, metrics_config: dict, analysis_types: list):
        """Run per-frame analysis using modular components."""
        print("Running per-frame analysis with modular components...")

        pck_df = self.data_processor.load_pck_per_frame_scores()
        if pck_df is None:
            print("Cannot proceed with per-frame analysis without data.")
            return

        combined_df = self.data_processor.process_per_frame_data(pck_df, metrics_config)

        if combined_df.empty:
            print("No combined data to analyze.")
            return

        # Run analyses with progress tracking
        progress = ProgressTracker(
            len(metrics_config) * len(analysis_types), "Running statistical analyses"
        )

        for metric_name in metrics_config.keys():
            self._run_statistical_analyses(
                combined_df, metric_name, analysis_types, progress
            )
            self._create_per_frame_visualizations(combined_df, metric_name)

        progress.finish()

        # Create PCK line plot if data is available
        self._create_pck_line_plot(combined_df)

        print(
            f"Per-frame analysis complete. Results saved to {self.config.save_folder}"
        )

    def _create_overall_visualizations(self, merged_df, all_metric_data, metric_name):
        """Create overall analysis visualizations using modular components."""
        viz_factory = VisualizationFactory()

        # Distribution plot
        dist_viz = viz_factory.create_visualizer("distribution", self.config)
        save_path = os.path.join(
            self.config.save_folder,
            f"overall_{metric_name}_distribution_{self.timestamp}.svg",
        )

        # Create a simple DataFrame for visualization
        viz_data = {metric_name: all_metric_data}
        import pandas as pd

        viz_df = pd.DataFrame(viz_data)

        dist_viz.create_plot(viz_df, metric_name, save_path)

    def _run_statistical_analyses(
        self, combined_df, metric_name, analysis_types, progress
    ):
        """Run statistical analyses using modular components."""
        for analysis_type in analysis_types:
            try:
                analyzer = AnalyzerFactory.create_analyzer(analysis_type, self.config)
                analyzer.analyze(combined_df, metric_name)
                progress.update()
            except ValueError as e:
                print(f"Warning: {e}")
                progress.update()

    def _create_per_frame_visualizations(self, combined_df, metric_name):
        """Create per-frame visualizations using modular components."""
        viz_factory = VisualizationFactory()

        # Scatter plot
        try:
            scatter_viz = viz_factory.create_visualizer("scatter", self.config)
            save_path = os.path.join(
                self.config.save_folder,
                f"per_frame_{metric_name}_scatter_{self.timestamp}.png",
            )
            scatter_viz.create_plot(combined_df, metric_name, save_path)
        except Exception as e:
            print(f"Warning: Could not create scatter plot for {metric_name}: {e}")

    def _create_pck_line_plot(self, combined_df):
        """Create PCK line plot for all thresholds using modular components."""
        viz_factory = VisualizationFactory()

        try:
            pck_viz = viz_factory.create_visualizer("pck_line", self.config)
            pck_viz.create_plot(combined_df, "pck_line", "")
        except ValueError as e:
            print(f"Warning: Could not create PCK line plot: {e}")


def main():
    """Main entry point for modular analysis."""
    dataset_name = "movi"

    metrics_config = {
        "brightness": "get_brightness_data",
        "contrast": "get_contrast_data",
    }

    run_overall_analysis = False
    run_per_frame_analysis = True
    per_frame_analysis_types = ["pck_frame_count"]

    # Test the modular components
    print("Testing modular components...")
    print(f"Available analyzers: {AnalyzerFactory.get_available_analyzers()}")
    print(f"Available visualizers: {VisualizationFactory.get_available_visualizers()}")
    print("Modular components test complete.\n")

    try:
        pipeline = ModularAnalysisPipeline(dataset_name)
        pipeline.run_complete_analysis(
            metrics_config=metrics_config,
            run_overall=run_overall_analysis,
            run_per_frame=run_per_frame_analysis,
            per_frame_analysis_types=per_frame_analysis_types,
        )
    except Exception as e:
        print(f"Error during analysis: {e}")
        raise


if __name__ == "__main__":
    main()

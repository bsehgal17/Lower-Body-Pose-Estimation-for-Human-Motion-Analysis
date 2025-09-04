"""
Analysis pipeline using separated components.
"""

from config import ConfigManager, load_analysis_config
from analyzers import AnalyzerFactory
from visualizers import VisualizationFactory
from data_processor import DataProcessor
from utils import PerformanceMonitor, ProgressTracker
import os
from datetime import datetime


class AnalysisPipeline:
    """Analysis pipeline using separated components."""

    def __init__(self, dataset_name: str):
        """Initialize the analysis pipeline."""
        self.dataset_name = dataset_name
        self.config = ConfigManager.load_config(dataset_name)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(self.config.save_folder, exist_ok=True)
        self.data_processor = DataProcessor(self.config)

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

        for metric_name, metric_data in results.items():
            self._create_overall_visualizations(
                metric_data["merged_df"], metric_data["all_metric_data"], metric_name
            )

        # Create PCK line plot for overall analysis if data is available
        if results:
            # Use the first available merged_df for PCK line plot
            first_result = next(iter(results.values()))
            if "merged_df" in first_result and first_result["merged_df"] is not None:
                self._create_pck_line_plot(
                    first_result["merged_df"], analysis_type="overall"
                )
                self._create_individual_pck_plots(
                    first_result["merged_df"], analysis_type="overall"
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
        self._create_pck_line_plot(combined_df, analysis_type="per_frame")

        # Create individual PCK threshold plots
        self._create_individual_pck_plots(combined_df, analysis_type="per_frame")

        print(
            f"Per-frame analysis complete. Results saved to {self.config.save_folder}"
        )

    def _create_overall_visualizations(self, merged_df, all_metric_data, metric_name):
        """Create overall analysis visualizations using separated components."""
        viz_factory = VisualizationFactory()

        # Distribution plots for aggregated metric data
        try:
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
        except Exception as e:
            print(
                f"Warning: Could not create distribution plots for {metric_name}: {e}"
            )

        # Bar plot for overall data
        try:
            bar_viz = viz_factory.create_visualizer("bar", self.config)
            save_path = os.path.join(
                self.config.save_folder,
                f"overall_{metric_name}_bar_{self.timestamp}.svg",
            )

            # Use the aggregated data for bar plot
            viz_data = {metric_name: all_metric_data}
            import pandas as pd

            viz_df = pd.DataFrame(viz_data)

            bar_viz.create_plot(viz_df, metric_name, save_path)
        except Exception as e:
            print(f"Warning: Could not create bar plot for {metric_name}: {e}")

        # Scatter plot using merged data if available
        if (
            merged_df is not None
            and not merged_df.empty
            and f"avg_{metric_name}" in merged_df.columns
        ):
            try:
                scatter_viz = viz_factory.create_visualizer("scatter", self.config)
                save_path = os.path.join(
                    self.config.save_folder,
                    f"overall_{metric_name}_scatter_{self.timestamp}.svg",
                )

                # Rename the column temporarily for scatter plot
                scatter_df = merged_df.copy()
                scatter_df[metric_name] = scatter_df[f"avg_{metric_name}"]

                scatter_viz.create_plot(scatter_df, metric_name, save_path)
            except Exception as e:
                print(f"Warning: Could not create scatter plot for {metric_name}: {e}")

    def _run_statistical_analyses(
        self, combined_df, metric_name, analysis_types, progress
    ):
        """Run statistical analyses using separated components."""
        for analysis_type in analysis_types:
            try:
                analyzer = AnalyzerFactory.create_analyzer(analysis_type, self.config)

                # Handle different analyzer signatures
                if analysis_type == "pck_brightness":
                    # PCK brightness analyzer works with DataFrame directly
                    results = analyzer.analyze(combined_df)

                    # Create visualizations for PCK brightness analysis
                    if results:
                        pck_brightness_viz = VisualizationFactory.create_visualizer(
                            "pck_brightness", self.config
                        )
                        save_path = f"per_frame_pck_brightness_{self.timestamp}"
                        pck_brightness_viz.create_plot(results, save_path)
                        pck_brightness_viz.create_combined_summary_plot(
                            results, save_path
                        )
                else:
                    # Other analyzers expect (data, metric_name)
                    analyzer.analyze(combined_df, metric_name)

                progress.update()
            except ValueError as e:
                print(f"Warning: {e}")
                progress.update()
            except Exception as e:
                print(f"Error running {analysis_type} analysis: {e}")
                progress.update()

    def _create_per_frame_visualizations(self, combined_df, metric_name):
        """Create per-frame visualizations using separated components."""
        viz_factory = VisualizationFactory()

        # Distribution plots (histogram and box plot)
        try:
            dist_viz = viz_factory.create_visualizer("distribution", self.config)
            save_path = os.path.join(
                self.config.save_folder,
                f"per_frame_{metric_name}_distribution_{self.timestamp}.svg",
            )
            dist_viz.create_plot(combined_df, metric_name, save_path)
        except Exception as e:
            print(
                f"Warning: Could not create distribution plots for {metric_name}: {e}"
            )

        # Scatter plot
        try:
            scatter_viz = viz_factory.create_visualizer("scatter", self.config)
            save_path = os.path.join(
                self.config.save_folder,
                f"per_frame_{metric_name}_scatter_{self.timestamp}.svg",
            )
            scatter_viz.create_plot(combined_df, metric_name, save_path)
        except Exception as e:
            print(f"Warning: Could not create scatter plot for {metric_name}: {e}")

        # Bar plot
        try:
            bar_viz = viz_factory.create_visualizer("bar", self.config)
            save_path = os.path.join(
                self.config.save_folder,
                f"per_frame_{metric_name}_bar_{self.timestamp}.svg",
            )
            bar_viz.create_plot(combined_df, metric_name, save_path)
        except Exception as e:
            print(f"Warning: Could not create bar plot for {metric_name}: {e}")

    def _create_pck_line_plot(self, combined_df, analysis_type="per_frame"):
        """Create PCK line plot for all thresholds using separated components."""
        viz_factory = VisualizationFactory()

        try:
            pck_viz = viz_factory.create_visualizer("pck_line", self.config)

            # Create a modified save path that includes the analysis type
            modified_save_path = f"{analysis_type}_pck_line_{self.timestamp}"

            pck_viz.create_plot(combined_df, "pck_line", modified_save_path)
            print(f"PCK line plot created for {analysis_type} analysis")

        except Exception as e:
            print(f"Warning: Could not create PCK line plot for {analysis_type}: {e}")
            import traceback

            traceback.print_exc()

    def _create_individual_pck_plots(self, combined_df, analysis_type="per_frame"):
        """Create individual line plots for each PCK threshold."""
        try:
            if not hasattr(self.config, "pck_per_frame_score_columns"):
                print("Warning: No PCK score columns found in config")
                return

            import matplotlib.pyplot as plt

            # Define bins (integers from -2 to 102)
            bins = list(range(-2, 102))

            for pck_col in self.config.pck_per_frame_score_columns:
                if pck_col not in combined_df.columns:
                    continue

                plt.figure(figsize=(10, 6))

                # Round values to nearest int and count
                counts = (
                    combined_df[pck_col]
                    .round()
                    .astype(int)
                    .value_counts()
                    .reindex(bins, fill_value=0)
                )

                plt.plot(
                    counts.index,
                    counts.values,
                    marker="o",
                    linestyle="-",
                    linewidth=2,
                    markersize=4,
                )

                # Labels and formatting
                plt.title(f"Frame Count per PCK Score - {pck_col}")
                plt.xlabel("PCK Score")
                plt.ylabel("Number of Frames")
                plt.grid(True, linestyle="--", alpha=0.6)

                # Save individual plot
                save_path = os.path.join(
                    self.config.save_folder,
                    f"{analysis_type}_{pck_col}_line_plot_{self.timestamp}.svg",
                )

                os.makedirs(self.config.save_folder, exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches="tight", format="svg")
                plt.close()

                print(f"Individual PCK line plot saved: {save_path}")

        except Exception as e:
            print(
                f"Warning: Could not create individual PCK plots for {analysis_type}: {e}"
            )
            import traceback

            traceback.print_exc()


def main():
    """Main entry point for analysis."""
    dataset_name = "movi"

    metrics_config = {
        "brightness": "get_brightness_data",
    }

    run_overall_analysis = True
    run_per_frame_analysis = True
    per_frame_analysis_types = AnalyzerFactory.get_available_analyzers()

    # Load analysis configuration from YAML
    analysis_config = load_analysis_config()

    # Test YAML configuration loading first
    print("🧪 Testing YAML-Based Dataset Configuration")
    print("=" * 60)

    datasets_to_test = ["humaneva", "movi"]

    for test_dataset in datasets_to_test:
        print(f"\n📋 Testing {test_dataset.upper()} configuration:")
        print("-" * 40)

        try:
            config = ConfigManager.load_config(test_dataset)

            print("✅ Configuration loaded successfully!")
            print(f"   Name: {config.name}")
            print(f"   Model: {config.model}")
            print(f"   PCK Overall columns: {config.pck_overall_score_columns}")
            print(f"   PCK Per-frame columns: {config.pck_per_frame_score_columns}")

            # Check for dataset-specific analysis config
            if hasattr(config, "analysis_config") and config.analysis_config:
                pck_brightness_config = config.analysis_config.get("pck_brightness", {})
                if pck_brightness_config:
                    score_groups = pck_brightness_config.get("score_groups", {})
                    default_group = pck_brightness_config.get(
                        "default_score_group", "all"
                    )
                    print(f"   Available score groups: {list(score_groups.keys())}")
                    print(f"   Default score group: {default_group}")

            validation_result = config.validate()
            print(
                f"   Configuration validation: {'✅ Passed' if validation_result else '❌ Failed'}"
            )

        except Exception as e:
            print(f"❌ Error loading {test_dataset} configuration: {e}")

    # Test the components
    print("\n" + "=" * 60)
    print("Testing analysis components...")
    print(f"Available analyzers: {AnalyzerFactory.get_available_analyzers()}")
    print(f"Available visualizers: {VisualizationFactory.get_available_visualizers()}")
    print(f"Will run per-frame analysis types: {per_frame_analysis_types}")
    print(
        f"Global analysis score groups: {analysis_config.get_available_score_groups()}"
    )
    print("Components test complete.\n")

    try:
        # Load dataset configuration for the main analysis
        dataset_config = ConfigManager.load_config(dataset_name)

        # Check if multi-analysis is enabled in config
        if analysis_config.is_multi_analysis_enabled():
            print("🚀 Running Multi-Analysis Pipeline from YAML Configuration")
            print("=" * 70)

            scenarios = analysis_config.get_multi_analysis_scenarios()

            for i, scenario in enumerate(scenarios, 1):
                scenario_name = scenario.get("name", f"scenario_{i}")
                score_group_name = scenario.get("score_group", "all")
                description = scenario.get("description", f"Analysis scenario {i}")

                # Try to get score group from dataset-specific config first, then global config
                score_group = None
                if (
                    hasattr(dataset_config, "analysis_config")
                    and dataset_config.analysis_config
                ):
                    pck_brightness_config = dataset_config.analysis_config.get(
                        "pck_brightness", {}
                    )
                    dataset_score_groups = pck_brightness_config.get("score_groups", {})
                    score_group = dataset_score_groups.get(score_group_name)

                # Fallback to global analysis config
                if score_group is None:
                    score_group = analysis_config.get_score_group(score_group_name)

                print(f"\n📊 Analysis {i}: {scenario_name}")
                print(f"Description: {description}")
                print(f"Score Group: {score_group_name} -> {score_group}")
                print("-" * 50)

                if i == 1:
                    # First analysis: Run complete pipeline
                    pipeline = AnalysisPipeline(dataset_name)
                    pipeline.run_complete_analysis(
                        metrics_config=metrics_config,
                        run_overall=run_overall_analysis,
                        run_per_frame=run_per_frame_analysis,
                        per_frame_analysis_types=per_frame_analysis_types,
                    )
                else:
                    # Subsequent analyses: Focus on PCK brightness with score filtering
                    pipeline = AnalysisPipeline(dataset_name)
                    pipeline.timestamp = pipeline.timestamp + f"_{scenario_name}"

                    # Load per-frame PCK data for filtered analysis
                    pck_df = pipeline.data_processor.load_pck_per_frame_scores()
                    if pck_df is not None:
                        combined_df = pipeline.data_processor.process_per_frame_data(
                            pck_df, metrics_config
                        )

                        if not combined_df.empty:
                            print(
                                f"Running PCK brightness analysis with score group: {score_group}"
                            )

                            try:
                                # Create analyzer with specific score groups from YAML
                                if score_group is not None:
                                    analyzer = AnalyzerFactory.create_analyzer(
                                        "pck_brightness",
                                        pipeline.config,
                                        score_groups=score_group,
                                    )
                                else:
                                    analyzer = AnalyzerFactory.create_analyzer(
                                        "pck_brightness", pipeline.config
                                    )

                                # Run analysis
                                results = analyzer.analyze(combined_df)

                                if (
                                    results
                                    and analysis_config.should_create_individual_plots()
                                ):
                                    # Create visualizations
                                    pck_brightness_viz = (
                                        VisualizationFactory.create_visualizer(
                                            "pck_brightness", pipeline.config
                                        )
                                    )
                                    save_path = f"per_frame_pck_brightness_{scenario_name}_{pipeline.timestamp}"
                                    pck_brightness_viz.create_plot(results, save_path)

                                    if analysis_config.should_create_combined_plots():
                                        pck_brightness_viz.create_combined_summary_plot(
                                            results, save_path
                                        )

                                    print(
                                        f"✅ PCK brightness analysis ({scenario_name}) completed successfully"
                                    )
                                else:
                                    print(
                                        f"❌ No results from {scenario_name} analysis"
                                    )

                            except Exception as e:
                                print(f"❌ Error in {scenario_name} analysis: {e}")
                        else:
                            print(
                                f"❌ No combined data available for {scenario_name} analysis"
                            )
                    else:
                        print(
                            f"❌ No per-frame PCK data available for {scenario_name} analysis"
                        )

                print(f"✅ {scenario_name} completed")

            print("\n" + "=" * 70)
            print("🎯 Multi-Analysis Summary:")
            for i, scenario in enumerate(scenarios, 1):
                scenario_name = scenario.get("name", f"scenario_{i}")
                print(f"   Analysis {i} ({scenario_name}): ✅ Completed")
            print("=" * 70)

        else:
            # Single analysis mode (original behavior)
            print("🚀 Running Single Analysis Pipeline")
            print("=" * 70)

            pipeline = AnalysisPipeline(dataset_name)
            pipeline.run_complete_analysis(
                metrics_config=metrics_config,
                run_overall=run_overall_analysis,
                run_per_frame=run_per_frame_analysis,
                per_frame_analysis_types=per_frame_analysis_types,
            )

            print("✅ Single analysis completed")

    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

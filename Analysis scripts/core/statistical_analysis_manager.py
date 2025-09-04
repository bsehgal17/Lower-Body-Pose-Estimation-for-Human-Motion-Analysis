"""
Statistical Analysis Manager Module

Handles the execution of statistical analyses on data.
"""

# from analyzers import AnalyzerFactory
from visualizers import VisualizationFactory
from utils import ProgressTracker


class StatisticalAnalysisManager:
    """Manages statistical analyses for different data types."""

    def __init__(self, config, timestamp=None):
        """Initialize the statistical analysis manager."""
        self.config = config
        self.timestamp = timestamp

    def run_statistical_analyses(
        self, combined_df, metric_name, analysis_types, progress=None
    ):
        """Run statistical analyses using separated components."""
        from analyzers import AnalyzerFactory

        if progress is None:
            progress = ProgressTracker(
                len(analysis_types), "Running statistical analyses"
            )

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

    def run_pck_brightness_analysis_with_score_groups(
        self, combined_df, score_group, scenario_name
    ):
        """Run PCK brightness analysis with specific score groups."""
        from analyzers import AnalyzerFactory

        try:
            # Create analyzer with specific score groups from YAML
            if score_group is not None:
                analyzer = AnalyzerFactory.create_analyzer(
                    "pck_brightness", self.config, score_groups=score_group
                )
            else:
                analyzer = AnalyzerFactory.create_analyzer(
                    "pck_brightness", self.config
                )

            # Run analysis
            results = analyzer.analyze(combined_df)

            if results:
                # Create visualizations
                pck_brightness_viz = VisualizationFactory.create_visualizer(
                    "pck_brightness", self.config
                )
                save_path = f"per_frame_pck_brightness_{scenario_name}_{self.timestamp}"
                pck_brightness_viz.create_plot(results, save_path)

                print(
                    f"✅ PCK brightness analysis ({scenario_name}) completed successfully"
                )
                return True
            else:
                print(f"❌ No results from {scenario_name} analysis")
                return False

        except Exception as e:
            print(f"❌ Error in {scenario_name} analysis: {e}")
            return False

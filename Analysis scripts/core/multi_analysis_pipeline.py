"""
Multi-Analysis Pipeline Module

Handles the execution of multiple analysis scenarios based on YAML configuration.
"""

# from analyzers import AnalyzerFactory
from .statistical_analysis_manager import StatisticalAnalysisManager


class MultiAnalysisPipeline:
    """Pipeline for running multiple analysis scenarios."""

    def __init__(self, config, data_processor, timestamp):
        """Initialize the multi-analysis pipeline."""
        self.config = config
        self.data_processor = data_processor
        self.timestamp = timestamp
        self.stats_manager = StatisticalAnalysisManager(config, timestamp)

    def run_multi_analysis(self, analysis_config, dataset_config, metrics_config):
        """Run multi-analysis pipeline from YAML configuration."""
        print("🚀 Running Multi-Analysis Pipeline from YAML Configuration")
        print("=" * 70)

        scenarios = analysis_config.get_multi_analysis_scenarios()
        results_summary = []

        for i, scenario in enumerate(scenarios, 1):
            scenario_name = scenario.get("name", f"scenario_{i}")
            score_group_name = scenario.get("score_group", "all")
            description = scenario.get("description", f"Analysis scenario {i}")

            # Try to get score group from dataset-specific config first, then global config
            score_group = self._get_score_group(
                dataset_config, analysis_config, score_group_name
            )

            print(f"\n📊 Analysis {i}: {scenario_name}")
            print(f"Description: {description}")
            print(f"Score Group: {score_group_name} -> {score_group}")
            print("-" * 50)

            if i == 1:
                # First analysis: Run complete pipeline
                from analyzers import AnalyzerFactory
                from .analysis_pipeline import AnalysisPipeline

                pipeline = AnalysisPipeline(self.config.name)
                success = pipeline.run_complete_analysis(
                    metrics_config=metrics_config,
                    run_overall=True,
                    run_per_frame=True,
                    per_frame_analysis_types=AnalyzerFactory.get_available_analyzers(),
                )
                results_summary.append((scenario_name, success))
            else:
                # Subsequent analyses: Focus on PCK brightness with score filtering
                success = self._run_filtered_pck_analysis(
                    scenario_name, score_group, metrics_config
                )
                results_summary.append((scenario_name, success))

            print(f"✅ {scenario_name} completed")

        self._print_multi_analysis_summary(scenarios, results_summary)
        return results_summary

    def _get_score_group(self, dataset_config, analysis_config, score_group_name):
        """Get score group from dataset-specific or global config."""
        score_group = None

        # Try dataset-specific config first
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

        return score_group

    def _run_filtered_pck_analysis(self, scenario_name, score_group, metrics_config):
        """Run PCK brightness analysis with score filtering."""
        # Create a new pipeline instance with modified timestamp
        from .analysis_pipeline import AnalysisPipeline

        pipeline = AnalysisPipeline(self.config.name)
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

                return self.stats_manager.run_pck_brightness_analysis_with_score_groups(
                    combined_df, score_group, scenario_name
                )
            else:
                print(f"❌ No combined data available for {scenario_name} analysis")
                return False
        else:
            print(f"❌ No per-frame PCK data available for {scenario_name} analysis")
            return False

    def _print_multi_analysis_summary(self, scenarios, results_summary):
        """Print summary of multi-analysis results."""
        print("\n" + "=" * 70)
        print("🎯 Multi-Analysis Summary:")
        for i, (scenario_name, success) in enumerate(results_summary, 1):
            status = "✅ Completed" if success else "❌ Failed"
            print(f"   Analysis {i} ({scenario_name}): {status}")
        print("=" * 70)

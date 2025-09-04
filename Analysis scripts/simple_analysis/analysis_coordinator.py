"""
Analysis Coordinator

Coordinates all the analysis scripts for easy usage.
This is the main entry point for the analysis system.
"""

import sys
import os
from typing import List

# Add the Analysis scripts directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pck_loader import PCKDataLoader
from brightness_analyzer import BrightnessAnalyzer
from plot_creator import PlotCreator
from statistical_analyzer import StatisticalAnalyzer
from overall_analyzer import OverallAnalyzer


class AnalysisCoordinator:
    """Coordinates all analysis components."""

    def __init__(self, dataset_name: str):
        """Initialize with dataset name."""
        self.dataset_name = dataset_name
        print(f"Initializing Analysis for {dataset_name.upper()}")
        print("=" * 60)

    def run_quick_analysis(self):
        """Run a quick analysis with the most common components."""
        print("Running Quick Analysis...")
        print("This includes: data preview + brightness analysis + basic plots")
        print("-" * 60)

        # 1. Load and preview data
        loader = PCKDataLoader(self.dataset_name)
        loader.preview_data("per_frame", 3)

        print("\n" + "-" * 30 + "\n")

        # 2. Run brightness analysis
        brightness_analyzer = BrightnessAnalyzer(self.dataset_name)
        brightness_analyzer.analyze_brightness_distribution()

        print("\n" + "-" * 30 + "\n")

        # 3. Create summary plots
        plot_creator = PlotCreator(self.dataset_name)
        plot_creator.create_summary_only("quick_analysis")

        print("\n✅ Quick analysis completed!")

    def run_complete_analysis(self):
        """Run complete analysis with all components."""
        print("Running Complete Analysis...")
        print("This includes: all data loading + all analyses + all visualizations")
        print("-" * 60)

        # 1. Data loading
        print("Step 1: Data Loading")
        loader = PCKDataLoader(self.dataset_name)
        overall_data = loader.load_overall_data()
        per_frame_data = loader.load_per_frame_data()

        if not per_frame_data:
            print("❌ Cannot proceed without per-frame data")
            return False

        print("\n" + "-" * 30 + "\n")

        # 2. Brightness analysis
        print("Step 2: Brightness Analysis")
        brightness_analyzer = BrightnessAnalyzer(self.dataset_name)
        brightness_analyzer.analyze_brightness_distribution()

        print("\n" + "-" * 30 + "\n")

        # 3. Statistical analysis
        print("Step 3: Statistical Analysis")
        stats_analyzer = StatisticalAnalyzer(self.dataset_name)
        stats_analyzer.run_all_statistical_analyses()

        print("\n" + "-" * 30 + "\n")

        # 4. Overall analysis (if overall data is available)
        if overall_data is not None:
            print("Step 4: Overall Analysis")
            overall_analyzer = OverallAnalyzer(self.dataset_name)
            overall_analyzer.run_overall_analysis()
        else:
            print("Step 4: Skipping overall analysis (no overall data)")

        print("\n" + "-" * 30 + "\n")

        # 5. Create all visualizations
        print("Step 5: Visualization")
        plot_creator = PlotCreator(self.dataset_name)
        plot_creator.create_all_plots("complete_analysis")

        print("\n✅ Complete analysis finished!")
        return True

    def run_custom_analysis(self, components: List[str]):
        """Run analysis with custom component selection."""
        print(f"Running Custom Analysis: {', '.join(components)}")
        print("-" * 60)

        available_components = {
            "data_preview": self._run_data_preview,
            "brightness": self._run_brightness_analysis,
            "statistics": self._run_statistical_analysis,
            "overall": self._run_overall_analysis,
            "plots": self._run_plotting,
            "plots_frequency": self._run_frequency_plots,
            "plots_summary": self._run_summary_plots,
        }

        for component in components:
            if component in available_components:
                print(f"\nRunning: {component}")
                print("-" * 30)
                available_components[component]()
            else:
                print(f"❌ Unknown component: {component}")

        print("\n✅ Custom analysis completed!")

    def _run_data_preview(self):
        """Run data preview component."""
        loader = PCKDataLoader(self.dataset_name)
        loader.preview_data("both", 5)

    def _run_brightness_analysis(self):
        """Run brightness analysis component."""
        analyzer = BrightnessAnalyzer(self.dataset_name)
        analyzer.analyze_brightness_distribution()

    def _run_statistical_analysis(self):
        """Run statistical analysis component."""
        analyzer = StatisticalAnalyzer(self.dataset_name)
        analyzer.run_all_statistical_analyses()

    def _run_overall_analysis(self):
        """Run overall analysis component."""
        analyzer = OverallAnalyzer(self.dataset_name)
        analyzer.run_overall_analysis()

    def _run_plotting(self):
        """Run all plotting component."""
        creator = PlotCreator(self.dataset_name)
        creator.create_all_plots("custom_analysis")

    def _run_frequency_plots(self):
        """Run frequency plots only."""
        creator = PlotCreator(self.dataset_name)
        creator.create_frequency_plots_only("custom_frequency")

    def _run_summary_plots(self):
        """Run summary plots only."""
        creator = PlotCreator(self.dataset_name)
        creator.create_summary_only("custom_summary")

    def list_available_components(self):
        """List all available analysis components."""
        print("Available Analysis Components:")
        print("=" * 40)
        print("1. data_preview - Preview PCK data")
        print("2. brightness - PCK brightness distribution analysis")
        print("3. statistics - Statistical analyses (ANOVA, bin analysis)")
        print("4. overall - Overall/video-level analysis")
        print("5. plots - All visualization plots")
        print("6. plots_frequency - Frequency distribution plots only")
        print("7. plots_summary - Summary plots only")
        print()
        print("Preset Analysis Types:")
        print("- quick: data_preview + brightness + plots_summary")
        print("- complete: all components")
        print("- custom: select specific components")

    def get_analysis_status(self):
        """Check what data and configurations are available."""
        print(f"Analysis Status for {self.dataset_name.upper()}:")
        print("=" * 50)

        try:
            # Check data availability
            loader = PCKDataLoader(self.dataset_name)

            overall_data = loader.load_overall_data()
            per_frame_data = loader.load_per_frame_data()

            print(
                f"Overall data: {'✅ Available' if overall_data is not None else '❌ Not available'}"
            )
            print(
                f"Per-frame data: {'✅ Available' if per_frame_data is not None else '❌ Not available'}"
            )

            if per_frame_data is not None:
                from config import ConfigManager

                config = ConfigManager.load_config(self.dataset_name)
                print(f"PCK thresholds: {config.pck_per_frame_score_columns}")

            # Check which analyses can be run
            print("\nPossible Analyses:")
            print(
                f"  Brightness analysis: {'✅' if per_frame_data is not None else '❌'}"
            )
            print(
                f"  Statistical analysis: {'✅' if per_frame_data is not None else '❌'}"
            )
            print(f"  Overall analysis: {'✅' if overall_data is not None else '❌'}")
            print(f"  Visualizations: {'✅' if per_frame_data is not None else '❌'}")

        except Exception as e:
            print(f"❌ Error checking status: {e}")

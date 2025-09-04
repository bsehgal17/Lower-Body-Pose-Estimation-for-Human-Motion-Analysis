"""
Simple Plot Creator Script

Creates visualizations from PCK brightness analysis results.
Focus: Just visualization, nothing else.
"""

import sys
import os

# Add the Analysis scripts directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import ConfigManager
from visualizers import VisualizationFactory
from simple_brightness_analyzer import SimpleBrightnessAnalyzer


class SimplePlotCreator:
    """Simple plot creator for PCK brightness analysis."""

    def __init__(self, dataset_name: str, score_groups=None):
        """Initialize with dataset name and optional score groups."""
        self.dataset_name = dataset_name
        self.config = ConfigManager.load_config(dataset_name)
        self.brightness_analyzer = SimpleBrightnessAnalyzer(dataset_name, score_groups)
        self.visualizer = VisualizationFactory.create_visualizer(
            "pck_brightness", self.config
        )


from config import ConfigManager
from visualizers import VisualizationFactory
from simple_brightness_analyzer import SimpleBrightnessAnalyzer


class SimplePlotCreator:
    """Simple plot creator for PCK brightness analysis."""

    def __init__(self, dataset_name: str):
        """Initialize with dataset name."""
        self.dataset_name = dataset_name
        self.config = ConfigManager.load_config(dataset_name)
        self.brightness_analyzer = SimpleBrightnessAnalyzer(dataset_name)
        self.visualizer = VisualizationFactory.create_visualizer(
            "pck_brightness", self.config
        )

    def create_all_plots(self, save_prefix: str = None) -> bool:
        """Create all available plots."""
        print(f"Creating plots for {self.dataset_name}...")
        print("=" * 50)

        # Get analysis results
        results = self.brightness_analyzer.analyze_brightness_distribution()
        if not results:
            print("❌ No analysis results available for plotting")
            return False

        # Set save prefix
        if save_prefix is None:
            save_prefix = f"simple_plots_{self.dataset_name}"

        try:
            # Create individual plots
            print("Creating individual threshold plots...")
            self.visualizer.create_plot(results, save_prefix)

            # Create combined summary
            print("Creating combined summary plot...")
            self.visualizer.create_combined_summary_plot(results, save_prefix)

            print("✅ All plots created successfully")
            print(f"   Save location: {self.config.save_folder}")
            return True

        except Exception as e:
            print(f"❌ Error creating plots: {e}")
            return False

    def create_frequency_plots_only(self, save_prefix: str = None) -> bool:
        """Create only the brightness frequency plots."""
        print("Creating brightness frequency plots only...")

        results = self.brightness_analyzer.analyze_brightness_distribution()
        if not results:
            return False

        if save_prefix is None:
            save_prefix = f"frequency_only_{self.dataset_name}"

        try:
            # Create only frequency plots by calling the specific method
            for pck_column, analysis_results in results.items():
                self.visualizer._create_brightness_frequency_plot(
                    analysis_results, pck_column, save_prefix
                )

            print("✅ Frequency plots created successfully")
            return True

        except Exception as e:
            print(f"❌ Error creating frequency plots: {e}")
            return False

    def create_statistics_plots_only(self, save_prefix: str = None) -> bool:
        """Create only the brightness statistics plots."""
        print("Creating brightness statistics plots only...")

        results = self.brightness_analyzer.analyze_brightness_distribution()
        if not results:
            return False

        if save_prefix is None:
            save_prefix = f"stats_only_{self.dataset_name}"

        try:
            # Create only statistics plots
            for pck_column, analysis_results in results.items():
                self.visualizer._create_brightness_statistics_plot(
                    analysis_results, pck_column, save_prefix
                )

            print("✅ Statistics plots created successfully")
            return True

        except Exception as e:
            print(f"❌ Error creating statistics plots: {e}")
            return False

    def create_frame_count_plots_only(self, save_prefix: str = None) -> bool:
        """Create only the frame count plots."""
        print("Creating frame count plots only...")

        results = self.brightness_analyzer.analyze_brightness_distribution()
        if not results:
            return False

        if save_prefix is None:
            save_prefix = f"frame_count_only_{self.dataset_name}"

        try:
            # Create only frame count plots
            for pck_column, analysis_results in results.items():
                self.visualizer._create_frame_count_plot(
                    analysis_results, pck_column, save_prefix
                )

            print("✅ Frame count plots created successfully")
            return True

        except Exception as e:
            print(f"❌ Error creating frame count plots: {e}")
            return False

    def create_summary_only(self, save_prefix: str = None) -> bool:
        """Create only the combined summary plot."""
        print("Creating summary plot only...")

        results = self.brightness_analyzer.analyze_brightness_distribution()
        if not results:
            return False

        if save_prefix is None:
            save_prefix = f"summary_only_{self.dataset_name}"

        try:
            self.visualizer.create_combined_summary_plot(results, save_prefix)
            print("✅ Summary plot created successfully")
            return True

        except Exception as e:
            print(f"❌ Error creating summary plot: {e}")
            return False

    def list_available_plots(self):
        """List all available plot types."""
        print("Available Plot Types:")
        print("-" * 30)
        print("1. all - All plots (frequency, statistics, frame counts, summary)")
        print("2. frequency - Brightness frequency distribution plots")
        print("3. statistics - Brightness statistics plots (mean, std, etc.)")
        print("4. frame_count - Frame count distribution plots")
        print("5. summary - Combined summary plot only")


def main():
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Simple Plot Creator")
    parser.add_argument("dataset", help="Dataset name (e.g., 'movi', 'humaneva')")
    parser.add_argument(
        "--type",
        choices=["all", "frequency", "statistics", "frame_count", "summary"],
        default="all",
        help="Type of plots to create",
    )
    parser.add_argument("--prefix", help="Save prefix for plot files")
    parser.add_argument("--list", action="store_true", help="List available plot types")

    args = parser.parse_args()

    try:
        creator = SimplePlotCreator(args.dataset)

        if args.list:
            creator.list_available_plots()
            return

        # Create plots based on type
        success = False
        if args.type == "all":
            success = creator.create_all_plots(args.prefix)
        elif args.type == "frequency":
            success = creator.create_frequency_plots_only(args.prefix)
        elif args.type == "statistics":
            success = creator.create_statistics_plots_only(args.prefix)
        elif args.type == "frame_count":
            success = creator.create_frame_count_plots_only(args.prefix)
        elif args.type == "summary":
            success = creator.create_summary_only(args.prefix)

        if success:
            print(f"\n✅ Plot creation completed for {args.dataset}")
        else:
            print(f"\n❌ Plot creation failed for {args.dataset}")
            sys.exit(1)

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

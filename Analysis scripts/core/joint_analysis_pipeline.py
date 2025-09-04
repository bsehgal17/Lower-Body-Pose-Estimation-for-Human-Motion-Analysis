"""
Joint Analysis Pipeline

Main orchestrator for modular joint analysis pipeline.
"""

from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

from core.joint_data_loader import JointDataLoader
from analyzers.joint_analyzer import JointAnalyzer
from visualizers.joint_visualizer import JointVisualizer
from processors.joint_report_generator import JointReportGenerator


class JointAnalysisPipeline:
    """Main pipeline orchestrator for joint analysis."""

    def __init__(
        self,
        dataset_name: str,
        joints_to_analyze: List[str],
        pck_thresholds: List[float],
        output_dir: str = None,
        save_results: bool = True,
    ):
        """Initialize the joint analysis pipeline.

        Args:
            dataset_name: Name of the dataset to analyze
            joints_to_analyze: List of joint names to analyze
            pck_thresholds: List of PCK thresholds to analyze
            output_dir: Output directory (auto-generated if None)
            save_results: Whether to save results to files
        """
        self.dataset_name = dataset_name
        self.joints_to_analyze = joints_to_analyze
        self.pck_thresholds = pck_thresholds
        self.save_results = save_results

        # Setup output directory
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = Path(
                f"./analysis_results/joint_analysis_{dataset_name}_{timestamp}"
            )

        # Initialize components
        self.data_loader = JointDataLoader(dataset_name)
        self.analyzer = JointAnalyzer(joints_to_analyze, pck_thresholds)
        self.visualizer = JointVisualizer(self.output_dir, save_results)
        self.report_generator = JointReportGenerator(
            self.output_dir, dataset_name, save_results
        )

    def run_complete_analysis(self) -> bool:
        """Run the complete joint analysis pipeline.

        Returns:
            bool: True if analysis completed successfully
        """
        print("Joint Analysis Pipeline")
        print("=" * 50)
        print(f"Dataset: {self.dataset_name}")
        print(f"Joints: {', '.join(self.joints_to_analyze)}")
        print(f"Thresholds: {self.pck_thresholds}")
        if self.save_results:
            print(f"Output: {self.output_dir}")
        print("=" * 50)

        try:
            # Step 1: Load and validate data
            print("\\n1. Loading and validating data...")
            pck_data = self.data_loader.load_and_validate(
                self.joints_to_analyze, self.pck_thresholds
            )

            if pck_data is None:
                print("ERROR: Failed to load data")
                return False

            print("✓ Data loading completed")

            # Step 2: Run analysis
            print("\\n2. Running joint analysis...")
            analysis_results = self.analyzer.run_complete_analysis(pck_data)

            if not analysis_results:
                print("ERROR: No analysis results generated")
                return False

            print("✓ Analysis completed")

            # Step 3: Generate visualizations
            if self.save_results:
                print("\\n3. Creating visualizations...")
                plot_data = self.analyzer.get_average_data_for_plotting(
                    analysis_results
                )
                self.visualizer.create_all_visualizations(plot_data)
                self.visualizer.create_summary_plot(plot_data)
                print("✓ Visualizations completed")

            # Step 4: Generate reports
            if self.save_results:
                print("\\n4. Generating reports...")
                self.report_generator.generate_all_reports(
                    analysis_results, self.joints_to_analyze, self.pck_thresholds
                )
                print("✓ Reports completed")

            print("\\n" + "=" * 50)
            print("Joint analysis pipeline completed successfully!")
            if self.save_results:
                print(f"Results saved to: {self.output_dir}")
            print("=" * 50)

            return True

        except Exception as e:
            print(f"\\nERROR: Pipeline failed: {e}")
            import traceback

            traceback.print_exc()
            return False

    def run_analysis_only(self) -> Dict[str, Any]:
        """Run only the analysis portion without visualizations or reports.

        Returns:
            dict: Analysis results or empty dict if failed
        """
        try:
            # Load and validate data
            pck_data = self.data_loader.load_and_validate(
                self.joints_to_analyze, self.pck_thresholds
            )

            if pck_data is None:
                return {}

            # Run analysis
            analysis_results = self.analyzer.run_complete_analysis(pck_data)
            return analysis_results

        except Exception as e:
            print(f"ERROR: Analysis failed: {e}")
            return {}

    def run_visualization_only(self, analysis_results: Dict[str, Any]) -> bool:
        """Run only visualization with provided analysis results.

        Args:
            analysis_results: Pre-computed analysis results

        Returns:
            bool: True if visualization completed successfully
        """
        try:
            if not analysis_results:
                print("ERROR: No analysis results provided")
                return False

            plot_data = self.analyzer.get_average_data_for_plotting(analysis_results)
            self.visualizer.create_all_visualizations(plot_data)
            self.visualizer.create_summary_plot(plot_data)

            return True

        except Exception as e:
            print(f"ERROR: Visualization failed: {e}")
            return False

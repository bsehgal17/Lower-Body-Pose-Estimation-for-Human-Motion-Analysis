"""
Main analysis pipeline using modular components.
"""
from modular_config import ConfigManager
from modular_data_processor import ModularDataProcessor
from modular_analyzers import AnalyzerFactory
from modular_visualizers import VisualizationFactory
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
    
    def run_complete_analysis(self, 
                             metrics_config: dict,
                             run_overall: bool,
                             run_per_frame: bool,
                             per_frame_analysis_types: list):
        """Run complete analysis pipeline."""
        print(f"Starting analysis pipeline for {self.dataset_name.upper()} dataset...")
        
        if run_overall:
            self._run_overall_analysis(metrics_config)
            print("\n" + "="*50 + "\n")
        
        if run_per_frame:
            self._run_per_frame_analysis(metrics_config, per_frame_analysis_types)
        
        print(f"\nComplete analysis pipeline finished for {self.dataset_name.upper()} dataset.")
    
    def _run_overall_analysis(self, metrics_config: dict):
        """Run overall analysis."""
        print("Running overall analysis...")
        
        processor = ModularDataProcessor(self.config)
        pck_df = processor.load_pck_scores()
        
        if pck_df is None:
            print("Cannot proceed with overall analysis without data.")
            return
        
        results = processor.process_overall_data(pck_df, metrics_config)
        
        for metric_name, metric_data in results.items():
            self._create_overall_visualizations(
                metric_data['merged_df'], 
                metric_data['all_metric_data'], 
                metric_name
            )
        
        print(f"Overall analysis complete. Results saved to {self.config.save_folder}")
    
    def _run_per_frame_analysis(self, metrics_config: dict, analysis_types: list):
        """Run per-frame analysis."""
        print("Running per-frame analysis...")
        
        processor = ModularDataProcessor(self.config)
        pck_df = processor.load_pck_per_frame_scores()
        
        if pck_df is None:
            print("Cannot proceed with per-frame analysis without data.")
            return
        
        combined_df = processor.process_per_frame_data(pck_df, metrics_config)
        
        if combined_df.empty:
            print("No combined data to analyze.")
            return
        
        for metric_name in metrics_config.keys():
            self._run_statistical_analyses(combined_df, metric_name, analysis_types)
            self._create_per_frame_visualizations(combined_df, metric_name)
        
        # Create PCK line plot if data is available
        self._create_pck_line_plot(combined_df)
        
        print(f"Per-frame analysis complete. Results saved to {self.config.save_folder}")
    
    def _create_overall_visualizations(self, merged_df, all_metric_data, metric_name):
        """Create overall analysis visualizations."""
        viz_factory = VisualizationFactory()
        
        # Distribution plot
        dist_viz = viz_factory.create_visualizer('distribution', self.config)
        dist_viz.create_plot(
            data=all_metric_data,
            metric_name=metric_name.title(),
            units='L* Channel, 0-255',
            title=f"Overall {metric_name.title()} Distribution ({self.config.name.upper()} {self.config.model})",
            save_path=os.path.join(self.config.save_folder, f'overall_{metric_name}_distribution_{self.timestamp}.svg')
        )
    
    def _run_statistical_analyses(self, combined_df, metric_name, analysis_types):
        """Run statistical analyses."""
        for analysis_type in analysis_types:
            try:
                analyzer = AnalyzerFactory.create_analyzer(analysis_type, self.config)
                analyzer.analyze(combined_df, metric_name)
            except ValueError as e:
                print(f"Warning: {e}")
    
    def _create_per_frame_visualizations(self, combined_df, metric_name):
        """Create per-frame visualizations."""
        pass  # Placeholder for per-frame visualizations
    
    def _create_pck_line_plot(self, combined_df):
        """Create PCK line plot for all thresholds."""
        viz_factory = VisualizationFactory()
        
        try:
            pck_viz = viz_factory.create_visualizer('pck_line', self.config)
            pck_viz.create_plot(combined_df, 'pck_line', '')
        except ValueError as e:
            print(f"Warning: Could not create PCK line plot: {e}")


def main():
    """Main entry point."""
    dataset_name = 'movi'
    
    metrics_config = {
        'brightness': 'get_brightness_data',
        'contrast': 'get_contrast_data'
    }
    
    run_overall_analysis = False
    run_per_frame_analysis = True
    per_frame_analysis_types = ['pck_frame_count']
    
    # Test the new modular components
    print("Testing new modular components...")
    from modular_analyzers import AnalyzerFactory
    from modular_visualizers import VisualizationFactory
    
    print(f"Available analyzers: {AnalyzerFactory.get_available_analyzers()}")
    print(f"Available visualizers: {VisualizationFactory.get_available_visualizers()}")
    print("Modular components test complete.\n")
    
    try:
        pipeline = AnalysisPipeline(dataset_name)
        pipeline.run_complete_analysis(
            metrics_config=metrics_config,
            run_overall=run_overall_analysis,
            run_per_frame=run_per_frame_analysis,
            per_frame_analysis_types=per_frame_analysis_types
        )
    except Exception as e:
        print(f"Error during analysis: {e}")
        raise


if __name__ == "__main__":
    main()

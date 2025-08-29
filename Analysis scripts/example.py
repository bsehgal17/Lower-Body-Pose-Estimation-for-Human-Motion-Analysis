"""
Example usage of the modular analysis system.
"""
from modular_config import ConfigManager
from modular_extractors import MetricExtractorFactory


def simple_brightness_analysis():
    """Simple example of brightness analysis."""
    print("=== Simple Brightness Analysis Example ===")
    
    # Load configuration
    config = ConfigManager.load_config('movi')
    print(f"Loaded config for {config.name} dataset")
    
    # Show available metrics
    available_metrics = MetricExtractorFactory.get_available_metrics()
    print(f"Available metrics: {available_metrics}")
    
    # Show available analyzers
    print("Available analyzers: anova, bin_analysis")
    
    print("Example completed successfully!")


def demonstrate_extensibility():
    """Show how easy it is to extend the system."""
    print("\n=== Extensibility Example ===")
    
    # Example custom metric
    from base_classes import BaseMetricExtractor
    
    class ExampleCustomMetric(BaseMetricExtractor):
        def extract(self):
            print(f"Would extract custom metric from {self.video_path}")
            return [1.0, 2.0, 3.0]  # Example data
    
    # Register it
    MetricExtractorFactory.register_extractor('custom_example', ExampleCustomMetric)
    
    # Now it's available
    updated_metrics = MetricExtractorFactory.get_available_metrics()
    print(f"Updated available metrics: {updated_metrics}")
    
    print("Extensibility demonstrated!")


if __name__ == "__main__":
    simple_brightness_analysis()
    demonstrate_extensibility()

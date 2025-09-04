"""
Factory for creating visualization instances.
"""

from core.base_classes import BaseVisualizer
from .distribution_visualizer import DistributionVisualizer
from .histogram_visualizer import HistogramVisualizer
from .boxplot_visualizer import BoxPlotVisualizer
from .scatter_visualizer import ScatterPlotVisualizer
from .bar_visualizer import BarPlotVisualizer
from .pck_brightness_visualizer import PCKBrightnessDistributionVisualizer


class VisualizationFactory:
    """Factory for creating visualization components."""

    _visualizers = {
        "distribution": DistributionVisualizer,
        "histogram": HistogramVisualizer,
        "boxplot": BoxPlotVisualizer,
        "scatter": ScatterPlotVisualizer,
        "bar": BarPlotVisualizer,
        "pck_brightness": PCKBrightnessDistributionVisualizer,
    }

    @classmethod
    def create_visualizer(cls, visualizer_type: str, config) -> BaseVisualizer:
        """Create a visualizer of the specified type."""
        visualizer_type = visualizer_type.lower()

        if visualizer_type not in cls._visualizers:
            raise ValueError(
                f"Unknown visualizer type: {visualizer_type}. Available: {list(cls._visualizers.keys())}"
            )

        return cls._visualizers[visualizer_type](config)

    @classmethod
    def register_visualizer(cls, visualizer_type: str, visualizer_class: type):
        """Register a new visualizer type."""
        if not issubclass(visualizer_class, BaseVisualizer):
            raise ValueError("Visualizer class must inherit from BaseVisualizer")

        cls._visualizers[visualizer_type.lower()] = visualizer_class

    @classmethod
    def get_available_visualizers(cls):
        """Get list of available visualizer types."""
        return list(cls._visualizers.keys())

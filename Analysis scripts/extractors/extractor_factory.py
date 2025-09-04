"""
Factory for creating metric extractor instances.
"""

from typing import List
from core.base_classes import BaseMetricExtractor
from .brightness_extractor import BrightnessExtractor
from .contrast_extractor import ContrastExtractor
from .sharpness_extractor import SharpnessExtractor


class MetricExtractorFactory:
    """Factory for creating metric extractors."""

    _extractors = {
        "brightness": BrightnessExtractor,
        "contrast": ContrastExtractor,
        "sharpness": SharpnessExtractor,
    }

    @classmethod
    def create_extractor(cls, metric_name: str, video_path: str) -> BaseMetricExtractor:
        """Create a metric extractor for the specified metric."""
        metric_name = metric_name.lower()

        if metric_name not in cls._extractors:
            raise ValueError(
                f"Unknown metric: {metric_name}. Available metrics: {list(cls._extractors.keys())}"
            )

        return cls._extractors[metric_name](video_path)

    @classmethod
    def register_extractor(cls, metric_name: str, extractor_class: type):
        """Register a new metric extractor."""
        if not issubclass(extractor_class, BaseMetricExtractor):
            raise ValueError("Extractor class must inherit from BaseMetricExtractor")

        cls._extractors[metric_name.lower()] = extractor_class

    @classmethod
    def get_available_metrics(cls) -> List[str]:
        """Get list of available metrics."""
        return list(cls._extractors.keys())

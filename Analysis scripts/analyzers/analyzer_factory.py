"""
Factory for creating analyzer instances.
"""

from base_classes import BaseAnalyzer
from .anova_analyzer import ANOVAAnalyzer
from .bin_analyzer import BinAnalyzer
from .pck_brightness_analyzer import PCKBrightnessAnalyzer


class AnalyzerFactory:
    """Factory for creating analyzers."""

    _analyzers = {
        "anova": ANOVAAnalyzer,
        "bin_analysis": BinAnalyzer,
        "pck_brightness": PCKBrightnessAnalyzer,
    }

    @classmethod
    def create_analyzer(cls, analyzer_type: str, config, **kwargs) -> BaseAnalyzer:
        """Create an analyzer of the specified type."""
        analyzer_type = analyzer_type.lower()

        if analyzer_type not in cls._analyzers:
            raise ValueError(
                f"Unknown analyzer type: {analyzer_type}. Available: {list(cls._analyzers.keys())}"
            )

        # Special handling for PCK brightness analyzer with score groups
        if analyzer_type == "pck_brightness" and "score_groups" in kwargs:
            return cls._analyzers[analyzer_type](
                config, score_groups=kwargs["score_groups"]
            )

        return cls._analyzers[analyzer_type](config)

    @classmethod
    def register_analyzer(cls, analyzer_type: str, analyzer_class: type):
        """Register a new analyzer type."""
        if not issubclass(analyzer_class, BaseAnalyzer):
            raise ValueError("Analyzer class must inherit from BaseAnalyzer")

        cls._analyzers[analyzer_type.lower()] = analyzer_class

    @classmethod
    def get_available_analyzers(cls):
        """Get list of available analyzer types."""
        return list(cls._analyzers.keys())

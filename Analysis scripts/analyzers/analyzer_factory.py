"""
Factory for creating analyzer instances.
"""

from core.base_classes import BaseAnalyzer


class AnalyzerFactory:
    """Factory for creating analyzers."""

    _analyzers = {
        "anova": "analyzers.anova_analyzer.ANOVAAnalyzer",
        "bin_analysis": "analyzers.bin_analyzer.BinAnalyzer",
        "pck_brightness": "analyzers.pck_brightness_analyzer.PCKBrightnessAnalyzer",
    }

    @classmethod
    def create_analyzer(cls, analyzer_type: str, config, **kwargs) -> BaseAnalyzer:
        """Create an analyzer of the specified type."""
        analyzer_type = analyzer_type.lower()

        if analyzer_type not in cls._analyzers:
            raise ValueError(
                f"Unknown analyzer type: {analyzer_type}. Available: {list(cls._analyzers.keys())}"
            )

        analyzer_path = cls._analyzers[analyzer_type]
        module_name, class_name = analyzer_path.rsplit(".", 1)

        # Dynamically import the module and get the class
        import importlib

        module = importlib.import_module(module_name)
        analyzer_class = getattr(module, class_name)

        # Special handling for PCK brightness analyzer with score groups
        if analyzer_type == "pck_brightness" and "score_groups" in kwargs:
            return analyzer_class(config, score_groups=kwargs["score_groups"])

        return analyzer_class(config)

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

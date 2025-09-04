"""
Analyzer components for statistical analysis.
"""

import sys


# Lazy imports to avoid circular dependencies
def _get_anova_analyzer():
    from .anova_analyzer import ANOVAAnalyzer

    return ANOVAAnalyzer


def _get_bin_analyzer():
    from .bin_analyzer import BinAnalyzer

    return BinAnalyzer


def _get_pck_brightness_analyzer():
    from .pck_brightness_analyzer import PCKBrightnessAnalyzer

    return PCKBrightnessAnalyzer


def _get_analyzer_factory():
    from .analyzer_factory import AnalyzerFactory

    return AnalyzerFactory


# Create lazy property accessors
module = sys.modules[__name__]


def __getattr__(name):
    if name == "ANOVAAnalyzer":
        return _get_anova_analyzer()
    elif name == "BinAnalyzer":
        return _get_bin_analyzer()
    elif name == "PCKBrightnessAnalyzer":
        return _get_pck_brightness_analyzer()
    elif name == "AnalyzerFactory":
        return _get_analyzer_factory()
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "ANOVAAnalyzer",
    "BinAnalyzer",
    "PCKBrightnessAnalyzer",
    "AnalyzerFactory",
]

"""
Core Module

Contains base classes and core functionality that is shared across
different analysis modules.
"""

import sys


# Lazy imports to avoid circular dependencies
def _get_data_processor():
    from .data_processor import DataProcessor

    return DataProcessor


def _get_brightness_extractor():
    from .brightness_extractor import BrightnessExtractor

    return BrightnessExtractor


def _get_config_managers():
    from .config_manager import AnalysisConfigManager, ConfigurationTester

    return AnalysisConfigManager, ConfigurationTester


def _get_visualization_manager():
    from .visualization_manager import VisualizationManager

    return VisualizationManager


def _get_statistical_analysis_manager():
    from .statistical_analysis_manager import StatisticalAnalysisManager

    return StatisticalAnalysisManager


def _get_multi_analysis_pipeline():
    from .multi_analysis_pipeline import MultiAnalysisPipeline

    return MultiAnalysisPipeline


def _get_analysis_pipeline():
    from .pipeline_manager import AnalysisPipeline

    return AnalysisPipeline


def __getattr__(name):
    if name == "DataProcessor":
        return _get_data_processor()
    elif name == "BrightnessExtractor":
        return _get_brightness_extractor()
    elif name == "AnalysisConfigManager":
        return _get_config_managers()[0]
    elif name == "ConfigurationTester":
        return _get_config_managers()[1]
    elif name == "VisualizationManager":
        return _get_visualization_manager()
    elif name == "StatisticalAnalysisManager":
        return _get_statistical_analysis_manager()
    elif name == "MultiAnalysisPipeline":
        return _get_multi_analysis_pipeline()
    elif name == "AnalysisPipeline":
        return _get_analysis_pipeline()
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "DataProcessor",
    "BrightnessExtractor",
    "AnalysisConfigManager",
    "ConfigurationTester",
    "VisualizationManager",
    "StatisticalAnalysisManager",
    "MultiAnalysisPipeline",
    "AnalysisPipeline",
]

"""
Core Module

Contains base classes and core functionality that is shared across
different analysis modules.
"""

from .data_processor import DataProcessor
from .brightness_extractor import BrightnessExtractor
from .config_manager import AnalysisConfigManager, ConfigurationTester
from .visualization_manager import VisualizationManager
from .statistical_analysis_manager import StatisticalAnalysisManager
from .multi_analysis_pipeline import MultiAnalysisPipeline
from .pipeline_manager import AnalysisPipeline

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

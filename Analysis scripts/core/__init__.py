"""
Core Module

Contains base classes and core functionality that is shared across
different analysis modules.
"""

from .data_processor import DataProcessor
from .brightness_extractor import BrightnessExtractor
from .config_manager import AnalysisConfigManager
from .visualization_manager import VisualizationManager
from .statistical_analysis_manager import StatisticalAnalysisManager
from .multi_analysis_pipeline import MultiAnalysisPipeline

__all__ = [
    "DataProcessor",
    "BrightnessExtractor",
    "AnalysisConfigManager",
    "VisualizationManager",
    "StatisticalAnalysisManager",
    "MultiAnalysisPipeline",
]

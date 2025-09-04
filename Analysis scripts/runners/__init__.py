"""
Runners Package

Contains modular runners for different types of analysis pipelines.
"""

from .single_analysis_runner import (
    run_single_analysis,
    run_single_analysis_with_options,
)
from .multi_analysis_runner import run_multi_analysis, run_custom_multi_analysis

__all__ = [
    "run_single_analysis",
    "run_single_analysis_with_options",
    "run_multi_analysis",
    "run_custom_multi_analysis",
]

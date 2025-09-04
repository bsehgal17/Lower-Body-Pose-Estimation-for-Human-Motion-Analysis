"""
PCK Analysis Module

Contains classes and functions specifically for PCK (Percentage of Correct Keypoints)
analysis, including PCK-brightness analysis and score filtering.
"""


# Lazy imports to avoid circular dependencies
def __getattr__(name):
    if name == "PCKBrightnessAnalysisPipeline":
        from .pck_brightness_analysis import PCKBrightnessAnalysisPipeline

        return PCKBrightnessAnalysisPipeline
    elif name == "PCKScoreFilter":
        from .pck_score_filter import PCKScoreFilter

        return PCKScoreFilter
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "PCKBrightnessAnalysisPipeline",
    "PCKScoreFilter",
]

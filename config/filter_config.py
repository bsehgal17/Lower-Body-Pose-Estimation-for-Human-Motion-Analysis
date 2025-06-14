from typing import Dict, Any, List
from dataclasses import dataclass


@dataclass
class OutlierRemovalConfig:
    enable: bool
    method: str  # Options: "iqr", "zscore"
    params: Dict[str, Any]


@dataclass
class FilterConfig:
    name: str
    params: Dict[str, Any]
    outlier_removal: OutlierRemovalConfig
    enable_interpolation: bool
    interpolation_kind: str
    enable_filter_plots: bool
    joints_to_filter: List[str]

    def __post_init__(self):
        # Convert dict to OutlierRemovalConfig if needed (for YAML loading)
        if isinstance(self.outlier_removal, dict):
            self.outlier_removal = OutlierRemovalConfig(**self.outlier_removal)

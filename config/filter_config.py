from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field


@dataclass
class OutlierRemovalConfig:
    enable: bool
    method: Optional[str] = None  # "iqr" or "zscore"
    params: Optional[Dict[str, Any]] = field(default_factory=dict)


@dataclass
class FilterConfig:
    name: str
    params: Optional[dict] = None
    input_dir: Optional[str] = None
    enable_interpolation: Optional[bool] = None
    interpolation_kind: Optional[str] = None
    enable_filter_plots: Optional[bool] = None
    joints_to_filter: Optional[List[str]] = None
    outlier_removal: Optional[OutlierRemovalConfig] = None

    def __post_init__(self):
        # Convert dict to OutlierRemovalConfig only if provided
        if isinstance(self.outlier_removal, dict):
            self.outlier_removal = OutlierRemovalConfig(**self.outlier_removal)

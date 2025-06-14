from typing import Dict, Any, List
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class OutlierRemovalConfig:
    enable: bool
    method: str  # Options: "iqr", "zscore"
    params: Dict[str, Any]


@dataclass
class FilterConfig:
    name: str
    params: Optional[dict] = field(default_factory=dict)
    input_dir: Optional[str] = None  # <-- ADD THIS
    enable_interpolation: bool = True
    interpolation_kind: str = "linear"
    enable_filter_plots: bool = False
    joints_to_filter: Optional[List[str]] = field(default_factory=list)
    outlier_removal: Optional[OutlierRemovalConfig] = field(
        default_factory=OutlierRemovalConfig)

    def __post_init__(self):
        # Convert dict to OutlierRemovalConfig if needed (for YAML loading)
        if isinstance(self.outlier_removal, dict):
            self.outlier_removal = OutlierRemovalConfig(**self.outlier_removal)

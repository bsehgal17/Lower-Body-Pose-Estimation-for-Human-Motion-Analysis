from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field


@dataclass
class OutlierRemovalConfig:
    enable: bool = False
    method: str = "iqr"  # Options: "iqr", "zscore"
    params: Dict[str, Any] = field(default_factory=lambda: {
        "iqr_multiplier": 1.5,
        # for zscore, use "z_threshold": 3.0
    })


@dataclass
class FilterConfig:
    name: str = "butterworth"
    params: Optional[Dict[str, Any]] = field(default_factory=dict)

    outlier_removal: OutlierRemovalConfig = field(
        default_factory=OutlierRemovalConfig)

    enable_interpolation: bool = True
    interpolation_kind: str = "linear"

    enable_filter_plots: bool = False

    joints_to_filter: List[str] = field(default_factory=lambda: [
        "LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_KNEE", "RIGHT_KNEE", "LEFT_HIP", "RIGHT_HIP"
    ])

    def __post_init__(self):
        # Convert dict to OutlierRemovalConfig if needed
        if isinstance(self.outlier_removal, dict):
            self.outlier_removal = OutlierRemovalConfig(**self.outlier_removal)

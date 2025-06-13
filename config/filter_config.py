from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field


@dataclass
class FilterConfig:
    name: str = "butterworth"  # e.g., 'gaussian', 'kalman', 'savitzky'
    params: Optional[Dict[str, Any]] = field(
        default_factory=dict)  # Filter-specific parameters

    enable_iqr: bool = False
    iqr_multiplier: float = 1.5

    enable_interpolation: bool = True
    interpolation_kind: str = "linear"

    enable_filter_plots: bool = False

    joints_to_filter: List[str] = field(
        default_factory=lambda: [
            "LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_KNEE", "RIGHT_KNEE", "LEFT_HIP", "RIGHT_HIP"
        ]
    )

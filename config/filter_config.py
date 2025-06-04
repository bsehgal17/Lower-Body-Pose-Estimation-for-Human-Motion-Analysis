from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class FilterConfig:
    method: str = "butterworth"
    cutoff: Optional[float] = 3.0
    order: Optional[int] = 2

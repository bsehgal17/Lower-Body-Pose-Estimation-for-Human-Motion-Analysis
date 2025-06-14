from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class AssessmentConfig:
    enabled: bool
    metric: str

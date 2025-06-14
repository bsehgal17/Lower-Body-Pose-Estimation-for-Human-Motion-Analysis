from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional


@dataclass
class EvaluationMetric:
    name: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationConfig:
    input_dir: Optional[str] = None  # ← directly here
    metrics: List[EvaluationMetric] = field(default_factory=list)

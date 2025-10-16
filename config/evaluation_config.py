from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional


class EvaluationMetric(BaseModel):
    name: str
    params: Dict[str, Any] = Field(default_factory=dict)


class EvaluationConfig(BaseModel):
    input_dir: Optional[str] = None  # ← directly here
    metrics: List[EvaluationMetric] = Field(default_factory=list)

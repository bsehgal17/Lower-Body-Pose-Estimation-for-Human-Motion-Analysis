from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field, model_validator


class OutlierRemovalConfig(BaseModel):
    enable: bool
    method: Optional[str]  # "iqr" or "zscore"
    params: Optional[Dict[str, Any]] = Field(default_factory=dict)


class FilterConfig(BaseModel):
    name: str
    params: Optional[dict]
    input_dir: Optional[str]
    enable_interpolation: Optional[bool]
    interpolation_kind: Optional[str]
    enable_filter_plots: Optional[bool]
    joints_to_filter: Optional[List[str]]
    outlier_removal: Optional[Union[OutlierRemovalConfig, dict]]

    @model_validator(mode="after")
    def convert_outlier_removal(self):
        # Convert dict to OutlierRemovalConfig only if provided
        if isinstance(self.outlier_removal, dict):
            self.outlier_removal = OutlierRemovalConfig(**self.outlier_removal)
        return self

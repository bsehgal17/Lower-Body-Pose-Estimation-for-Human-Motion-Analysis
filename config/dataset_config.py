from pydantic import BaseModel
from typing import Optional, Dict


class DatasetConfig(BaseModel):
    keypoint_format: Optional[str] = None  # e.g. "utils.joint_enum"
    joint_enum_module: Optional[str] = None  # e.g. "utils.joint_enum"
    sync_data: Optional[Dict[str, Dict[str, list]]] = (
        None  # e.g. {"S1": {"Walking": [10, 12, 14]}}
    )

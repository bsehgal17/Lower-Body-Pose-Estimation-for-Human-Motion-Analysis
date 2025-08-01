from dataclasses import dataclass
from typing import Optional, Dict


@dataclass
class DatasetConfig:
    keypoint_format: Optional[str] = None  # e.g. "utils.joint_enum"
    joint_enum_module: Optional[str] = None  # e.g. "utils.joint_enum"
    sync_data: Optional[Dict[str, Dict[str, list]]] = (
        None  # e.g. {"S1": {"Walking": [10, 12, 14]}}
    )

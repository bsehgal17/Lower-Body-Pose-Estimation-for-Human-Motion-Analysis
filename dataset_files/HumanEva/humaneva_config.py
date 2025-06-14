from dataclasses import dataclass
from typing import Dict, List


@dataclass
class SyncDataConfig:
    """
    Sync frame information for HumanEva subjects, actions, and cameras.
    Expected structure: subject -> action -> list of sync frame indices per camera.
    """

    data: Dict[str, Dict[str, List[int]]]

from typing import Dict, Tuple
from dataclasses import dataclass


@dataclass
class SyncDataConfig:
    """Configuration for synchronization data (e.g., HumanEva dataset)."""

    data: Dict[str, Dict[str, Tuple[int, int, int]]]

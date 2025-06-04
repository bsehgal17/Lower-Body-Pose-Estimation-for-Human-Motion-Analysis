from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class SyncDataConfig:
    """Configuration for synchronization data (e.g., HumanEva dataset)."""

    data: Dict[str, Dict[str, Tuple[int, int, int]]] = field(
        default_factory=lambda: {
            "S1": {"Walking 1": (667, 667, 667), "Jog 1": (49, 50, 51)},
            "S2": {"Walking 1": (547, 547, 546), "Jog 1": (493, 491, 502)},
            "S3": {"Walking 1": (524, 524, 524), "Jog 1": (464, 462, 462)},
        }
    )

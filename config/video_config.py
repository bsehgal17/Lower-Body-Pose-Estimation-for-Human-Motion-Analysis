from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class VideoConfig:
    """Configuration for video processing."""

    extensions: Tuple[str, ...]
    # Add more video specific settings here, e.g.,
    # target_fps: Optional[int] = None
    # resize_dim: Optional[Tuple[int, int]] = None

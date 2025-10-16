from typing import Tuple
from pydantic import BaseModel


class VideoConfig(BaseModel):
    """Configuration for video processing."""

    extensions: Tuple[str, ...]
    # Add more video specific settings here, e.g.,
    # target_fps: Optional[int] = None
    # resize_dim: Optional[Tuple[int, int]] = None

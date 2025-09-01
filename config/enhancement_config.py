from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class CLAHEConfig:
    """Configuration for CLAHE (Contrast Limited Adaptive Histogram Equalization) enhancement."""

    apply_clahe: bool
    clip_limit: float
    tile_grid_size: Tuple[int, int]
    color_space: str
    input_dir: Optional[str]
    output_dir: Optional[str]
    file_extensions: Tuple[str, ...]
    batch_process: bool

from typing import Optional, List
from dataclasses import dataclass


@dataclass
class CLAHEConfig:
    """Configuration for CLAHE (Contrast Limited Adaptive Histogram Equalization) enhancement."""

    clip_limit: float
    tile_grid_size: List[int]
    color_space: str = "LAB"


@dataclass
class BrightnessConfig:
    """Configuration for brightness adjustment."""

    factor: float


@dataclass
class BlurConfig:
    """Configuration for Gaussian blur."""

    kernel_size: List[int]
    sigma_x: float = 0
    sigma_y: float = 0


@dataclass
class EnhancementProcessingConfig:
    """Configuration for enhancement processing settings."""

    batch_size: int = 10
    dataset_structure: bool = True
    generate_report: bool = True


@dataclass
class EnhancementConfig:
    """Main configuration for video enhancement."""

    type: str
    clahe: Optional[CLAHEConfig] = None
    brightness: Optional[BrightnessConfig] = None
    blur: Optional[BlurConfig] = None
    processing: Optional[EnhancementProcessingConfig] = None

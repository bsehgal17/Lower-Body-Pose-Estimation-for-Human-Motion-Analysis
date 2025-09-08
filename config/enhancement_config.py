from typing import Optional, List, Dict, Any
from dataclasses import dataclass


@dataclass
class CLAHEConfig:
    """Configuration for CLAHE (Contrast Limited Adaptive Histogram Equalization) enhancement."""

    clip_limit: float
    tile_grid_size: List[int]
    color_space: str = "LAB"


@dataclass
class FilteredCLAHEConfig:
    """Configuration for CLAHE enhancement with image filtering."""

    clip_limit: float
    tile_grid_size: List[int]
    color_space: str = "HSV"
    filters: List[str] = None
    filter_params: Dict[str, Any] = None

    def __post_init__(self):
        if self.filters is None:
            self.filters = ["bilateral"]
        if self.filter_params is None:
            self.filter_params = {}


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
class GammaConfig:
    """Configuration for gamma correction enhancement."""

    gamma: float
    color_space: str = "BGR"
    input_dir: Optional[str] = None
    output_dir: Optional[str] = None
    file_extensions: List[str] = None

    def __post_init__(self):
        if self.file_extensions is None:
            self.file_extensions = [".mp4", ".avi", ".mov", ".mkv"]


@dataclass
class FilteredGammaConfig:
    """Configuration for gamma correction enhancement with image filtering."""

    gamma: float
    color_space: str = "HSV"
    filters: List[str] = None
    filter_params: Dict[str, Any] = None

    def __post_init__(self):
        if self.filters is None:
            self.filters = ["bilateral"]
        if self.filter_params is None:
            self.filter_params = {}


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
    filtered_clahe: Optional[FilteredCLAHEConfig] = None
    brightness: Optional[BrightnessConfig] = None
    blur: Optional[BlurConfig] = None
    gamma: Optional[GammaConfig] = None
    filtered_gamma: Optional[FilteredGammaConfig] = None
    processing: Optional[EnhancementProcessingConfig] = None
    create_comparison_images: Optional[bool] = None

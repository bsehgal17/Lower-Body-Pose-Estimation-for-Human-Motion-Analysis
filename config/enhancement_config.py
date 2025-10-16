from typing import Optional, List, Dict, Any
from pydantic import BaseModel, model_validator


class CLAHEConfig(BaseModel):
    """Configuration for CLAHE (Contrast Limited Adaptive Histogram Equalization) enhancement."""

    clip_limit: float
    tile_grid_size: List[int]
    color_space: str


class FilteredCLAHEConfig(BaseModel):
    """Configuration for CLAHE enhancement with image filtering."""

    clip_limit: float
    tile_grid_size: List[int]
    color_space: str
    filters: Optional[List[str]]
    filter_params: Optional[Dict[str, Any]]

    @model_validator(mode="after")
    def set_defaults(self):
        if self.filters is None:
            self.filters = ["bilateral"]
        if self.filter_params is None:
            self.filter_params = {}
        return self


class BrightnessConfig(BaseModel):
    """Configuration for brightness adjustment."""

    factor: float


class BlurConfig(BaseModel):
    """Configuration for Gaussian blur."""

    kernel_size: List[int]
    sigma_x: float
    sigma_y: float


class GammaConfig(BaseModel):
    """Configuration for gamma correction enhancement."""

    gamma: float
    color_space: str
    input_dir: Optional[str]
    output_dir: Optional[str]
    file_extensions: Optional[List[str]]

    @model_validator(mode="after")
    def set_defaults(self):
        if self.file_extensions is None:
            self.file_extensions = [".mp4", ".avi", ".mov", ".mkv"]
        return self


class FilteredGammaConfig(BaseModel):
    """Configuration for gamma correction enhancement with image filtering."""

    gamma: float
    color_space: str
    filters: Optional[List[str]]
    filter_params: Optional[Dict[str, Any]]

    @model_validator(mode="after")
    def set_defaults(self):
        if self.filters is None:
            self.filters = ["bilateral"]
        if self.filter_params is None:
            self.filter_params = {}
        return self


class EnhancementProcessingConfig(BaseModel):
    """Configuration for enhancement processing settings."""

    batch_size: int = 10
    dataset_structure: bool = True
    generate_report: bool = True


class EnhancementConfig(BaseModel):
    """Main configuration for video enhancement."""

    type: str
    clahe: Optional[CLAHEConfig]
    filtered_clahe: Optional[FilteredCLAHEConfig]
    brightness: Optional[BrightnessConfig]
    blur: Optional[BlurConfig]
    gamma: Optional[GammaConfig]
    filtered_gamma: Optional[FilteredGammaConfig]
    processing: Optional[EnhancementProcessingConfig]
    create_comparison_images: Optional[bool]

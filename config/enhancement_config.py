from typing import Optional, List, Dict, Any
from pydantic import BaseModel, model_validator, Field


class CLAHEConfig(BaseModel):
    """
    Configuration for CLAHE (Contrast Limited Adaptive Histogram Equalization) enhancement.

    CLAHE improves local contrast in images by applying histogram equalization
    to small regions (tiles) rather than the entire image, with clipping to prevent over-amplification.
    """

    clip_limit: float = Field(
        ...,
        gt=0.0,
        description="Threshold for contrast limiting in CLAHE algorithm. "
        "Higher values allow more contrast enhancement but may introduce artifacts. "
        "Typical range: 1.0-4.0. Values >4.0 may cause over-enhancement.",
    )

    tile_grid_size: List[int] = Field(
        ...,
        min_items=2,
        max_items=2,
        description="Size of the contextual regions (tiles) for adaptive histogram equalization. "
        "Format: [width, height] in pixels. Smaller tiles = more local adaptation "
        "but may cause blocking artifacts. Typical values: [8, 8] to [64, 64].",
    )

    color_space: str = Field(
        default="LAB",
        description="Color space for CLAHE processing. Options: 'LAB', 'HSV', 'BGR', 'YUV'. "
        "'LAB' applies CLAHE to L channel (luminance), preserving color information. "
        "'HSV' applies to V channel (brightness). 'BGR' processes all channels.",
    )


class FilteredCLAHEConfig(BaseModel):
    """
    Configuration for CLAHE enhancement with pre-processing image filtering.

    Applies image filters before CLAHE to reduce noise and improve enhancement quality.
    """

    clip_limit: float = Field(
        ...,
        gt=0.0,
        description="CLAHE clip limit threshold. See CLAHEConfig.clip_limit for details.",
    )

    tile_grid_size: List[int] = Field(
        ...,
        min_items=2,
        max_items=2,
        description="CLAHE tile grid size. See CLAHEConfig.tile_grid_size for details.",
    )

    color_space: str = Field(
        default="HSV",
        description="Color space for processing. See CLAHEConfig.color_space for options.",
    )

    filters: Optional[List[str]] = Field(
        default=None,
        description="List of image filters to apply before CLAHE. Options: 'bilateral', "
        "'gaussian', 'median', 'morphological'. Filters are applied in order. "
        "Default: ['bilateral'] for noise reduction while preserving edges.",
    )

    filter_params: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Parameters for each filter. Keys match filter names, values are parameter dicts. "
        "Example: {'bilateral': {'d': 9, 'sigmaColor': 75, 'sigmaSpace': 75}}",
    )

    @model_validator(mode="after")
    def set_defaults(self):
        if self.filters is None:
            self.filters = ["bilateral"]
        if self.filter_params is None:
            self.filter_params = {}
        return self


class BrightnessConfig(BaseModel):
    """
    Configuration for linear brightness adjustment.

    Applies uniform brightness scaling to all pixels in the image.
    """

    factor: float = Field(
        ...,
        gt=0.0,
        description="Brightness multiplication factor. Values >1.0 increase brightness, "
        "values <1.0 decrease brightness. Factor of 1.0 = no change. "
        "Typical range: 0.5-2.0. Values >2.0 may cause overexposure.",
    )


class BlurConfig(BaseModel):
    """
    Configuration for Gaussian blur filtering.

    Applies Gaussian smoothing to reduce noise and fine details in images.
    """

    kernel_size: List[int] = Field(
        ...,
        min_items=2,
        max_items=2,
        description="Gaussian kernel size [width, height] in pixels. Must be odd numbers. "
        "Larger kernels = stronger blur effect. Typical values: [3, 3] to [21, 21]. "
        "Use square kernels [n, n] for symmetric blur.",
    )

    sigma_x: float = Field(
        default=0,
        ge=0.0,
        description="Standard deviation in X direction for Gaussian kernel. "
        "Controls blur strength horizontally. Value of 0 auto-calculates from kernel size. "
        "Larger values = stronger blur in X direction.",
    )

    sigma_y: float = Field(
        default=0,
        ge=0.0,
        description="Standard deviation in Y direction for Gaussian kernel. "
        "Controls blur strength vertically. Value of 0 auto-calculates from kernel size. "
        "Larger values = stronger blur in Y direction.",
    )


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

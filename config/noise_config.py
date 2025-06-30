from pydantic import BaseModel, model_validator
from typing import Optional, Tuple


class NoiseConfig(BaseModel):
    apply_poisson_noise: bool
    poisson_scale: Optional[float] = None

    apply_gaussian_noise: bool
    gaussian_std: Optional[float] = None

    apply_motion_blur: bool
    motion_blur_kernel_size: Optional[int] = None

    apply_brightness_reduction: bool
    brightness_factor: Optional[int] = None

    target_resolution: Optional[Tuple[int, int]] = None

    @model_validator(mode="after")
    def validate_dependencies(self) -> "NoiseConfig":
        if self.apply_poisson_noise and self.poisson_scale is None:
            raise ValueError(
                "`poisson_scale` must be set when `apply_poisson_noise` is True.")
        if self.apply_gaussian_noise and self.gaussian_std is None:
            raise ValueError(
                "`gaussian_std` must be set when `apply_gaussian_noise` is True.")
        if self.apply_motion_blur and self.motion_blur_kernel_size is None:
            raise ValueError(
                "`motion_blur_kernel_size` must be set when `apply_motion_blur` is True.")
        if self.apply_brightness_reduction and self.brightness_factor is None:
            raise ValueError(
                "`brightness_factor` must be set when `apply_brightness_reduction` is True.")
        return self

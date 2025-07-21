from pydantic import BaseModel, model_validator
from typing import Optional, Tuple


class NoiseConfig(BaseModel):
    apply_poisson_noise: Optional[bool] = None
    poisson_scale: Optional[float] = None

    apply_gaussian_noise: Optional[bool] = None
    gaussian_std: Optional[float] = None

    apply_motion_blur: Optional[bool] = None
    motion_blur_kernel_size: Optional[int] = None

    apply_brightness_reduction: Optional[bool] = None
    brightness_factor: Optional[float] = None

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

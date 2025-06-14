from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class NoiseConfig:
    apply_poisson_noise: bool
    poisson_scale: float

    apply_gaussian_noise: bool
    gaussian_std: float

    apply_motion_blur: bool
    motion_blur_kernel_size: int

    apply_brightness_reduction: bool
    brightness_factor: int

    target_resolution: Optional[Tuple[int, int]]

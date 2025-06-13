from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class NoiseConfig:
    apply_poisson_noise: bool = True
    poisson_scale: float = 1.0

    apply_gaussian_noise: bool = True
    gaussian_std: float = 5.0

    apply_motion_blur: bool = True
    motion_blur_kernel_size: int = 5

    apply_brightness_reduction: bool = False
    brightness_factor: int = 40

    target_resolution: Optional[Tuple[int, int]] = None

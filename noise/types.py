import numpy as np
import cv2
import logging

logger = logging.getLogger(__name__)


def add_realistic_noise(
    image: np.ndarray, poisson_scale: float = 1.0, gaussian_std: float = 5.0
) -> np.ndarray:
    # Poisson + Gaussian noise
    noisy_poisson = (
        np.random.poisson(image.astype(np.float32) *
                          poisson_scale) / poisson_scale
    )
    noisy_poisson = np.clip(noisy_poisson, 0, 255).astype(np.uint8)
    gaussian_noise = np.random.normal(
        0, gaussian_std, image.shape).astype(np.int16)
    noisy_combined = cv2.add(
        noisy_poisson.astype(np.int16), gaussian_noise, dtype=cv2.CV_16S
    )
    return np.clip(noisy_combined, 0, 255).astype(np.uint8)


def apply_motion_blur(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    if kernel_size % 2 == 0:
        kernel_size += 1
        logger.warning(
            f"Motion blur kernel size adjusted to odd: {kernel_size}")
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[kernel_size // 2, :] = np.ones(kernel_size)
    return cv2.filter2D(image, -1, kernel / kernel_size)

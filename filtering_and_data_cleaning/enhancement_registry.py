"""
Enhancement Registry for Video Processing

This module provides a registry of enhancement techniques that can be applied to videos,
similar to the filter registry pattern used for keypoint filtering.
"""

import logging
from typing import Dict, Callable, Union
from pathlib import Path
import cv2
import numpy as np

from .clahe_enhancement import CLAHEEnhancer
from .gamma_enhancement import GammaEnhancer

logger = logging.getLogger(__name__)


def apply_clahe_enhancement(
    video_path: Union[str, Path], output_path: Union[str, Path], **kwargs
) -> bool:
    """
    Apply CLAHE enhancement to a video file.

    Args:
        video_path: Path to input video
        output_path: Path to output enhanced video
        **kwargs: CLAHE parameters (clip_limit, tile_grid_size, color_space)

    Returns:
        bool: True if successful, False otherwise
    """
    clip_limit = kwargs.get("clip_limit")
    tile_grid_size = kwargs.get("tile_grid_size")
    color_space = kwargs.get("color_space", "LAB")

    if clip_limit is None or tile_grid_size is None:
        raise ValueError(
            "CLAHE enhancement requires clip_limit and tile_grid_size parameters"
        )

    enhancer = CLAHEEnhancer(
        clip_limit=clip_limit, tile_grid_size=tuple(tile_grid_size)
    )

    return enhancer.enhance_video(
        input_path=Path(video_path),
        output_path=Path(output_path),
        color_space=color_space,
    )


def apply_histogram_equalization(
    video_path: Union[str, Path], output_path: Union[str, Path], **kwargs
) -> bool:
    """
    Apply global histogram equalization to a video file.

    Args:
        video_path: Path to input video
        output_path: Path to output enhanced video
        **kwargs: Additional parameters (currently unused)

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return False

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Apply histogram equalization
            # Convert to YUV color space
            yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
            # Apply histogram equalization to Y channel
            yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
            # Convert back to BGR
            enhanced_frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

            out.write(enhanced_frame)
            frame_count += 1

        cap.release()
        out.release()

        logger.debug(f"Processed {frame_count} frames for {Path(video_path).name}")
        return True

    except Exception as e:
        logger.error(f"Histogram equalization failed for {video_path}: {e}")
        return False


def apply_gaussian_blur(
    video_path: Union[str, Path], output_path: Union[str, Path], **kwargs
) -> bool:
    """
    Apply Gaussian blur to a video file.

    Args:
        video_path: Path to input video
        output_path: Path to output enhanced video
        **kwargs: blur parameters (kernel_size, sigma_x, sigma_y)

    Returns:
        bool: True if successful, False otherwise
    """
    kernel_size = kwargs.get("kernel_size", (5, 5))
    sigma_x = kwargs.get("sigma_x", 0)
    sigma_y = kwargs.get("sigma_y", 0)

    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return False

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Apply Gaussian blur
            blurred_frame = cv2.GaussianBlur(frame, kernel_size, sigma_x, sigma_y)

            out.write(blurred_frame)
            frame_count += 1

        cap.release()
        out.release()

        logger.debug(
            f"Applied Gaussian blur to {frame_count} frames for {Path(video_path).name}"
        )
        return True

    except Exception as e:
        logger.error(f"Gaussian blur failed for {video_path}: {e}")
        return False


def apply_brightness_adjustment(
    video_path: Union[str, Path], output_path: Union[str, Path], **kwargs
) -> bool:
    """
    Apply brightness adjustment to a video file.

    Args:
        video_path: Path to input video
        output_path: Path to output enhanced video
        **kwargs: adjustment parameters (factor)

    Returns:
        bool: True if successful, False otherwise
    """
    factor = kwargs.get("factor")
    if factor is None:
        raise ValueError("Brightness adjustment requires factor parameter")

    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return False

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Apply brightness adjustment
            # Convert to float to avoid overflow
            frame_float = frame.astype(np.float32)
            adjusted_frame = frame_float * factor
            # Clip to valid range and convert back to uint8
            adjusted_frame = np.clip(adjusted_frame, 0, 255).astype(np.uint8)

            out.write(adjusted_frame)
            frame_count += 1

        cap.release()
        out.release()

        logger.debug(
            f"Applied brightness adjustment to {frame_count} frames for {Path(video_path).name}"
        )
        return True

    except Exception as e:
        logger.error(f"Brightness adjustment failed for {video_path}: {e}")
        return False


def apply_gamma_correction(
    video_path: Union[str, Path], output_path: Union[str, Path], **kwargs
) -> bool:
    """
    Apply gamma correction to a video file.

    Args:
        video_path: Path to input video
        output_path: Path to output enhanced video
        **kwargs: gamma parameters (gamma, color_space)

    Returns:
        bool: True if successful, False otherwise
    """
    gamma = kwargs.get("gamma")
    color_space = kwargs.get("color_space", "BGR")

    if gamma is None:
        raise ValueError("Gamma correction requires gamma parameter")

    enhancer = GammaEnhancer(gamma=gamma)

    return enhancer.enhance_video(
        input_path=Path(video_path),
        output_path=Path(output_path),
        color_space=color_space,
    )


# Enhancement function registry
ENHANCEMENT_FN_MAP: Dict[str, Callable] = {
    "clahe": apply_clahe_enhancement,
    "histogram_eq": apply_histogram_equalization,
    "gaussian_blur": apply_gaussian_blur,
    "brightness_adjustment": apply_brightness_adjustment,
    "gamma_correction": apply_gamma_correction,
}


def get_enhancement_function(enhancement_type: str) -> Callable:
    """
    Get enhancement function by name.

    Args:
        enhancement_type: Name of the enhancement technique

    Returns:
        Callable: Enhancement function

    Raises:
        ValueError: If enhancement type is not found
    """
    if enhancement_type not in ENHANCEMENT_FN_MAP:
        available_types = list(ENHANCEMENT_FN_MAP.keys())
        raise ValueError(
            f"Unknown enhancement type: {enhancement_type}. Available: {available_types}"
        )

    return ENHANCEMENT_FN_MAP[enhancement_type]


def register_enhancement_function(name: str, func: Callable):
    """
    Register a new enhancement function.

    Args:
        name: Name of the enhancement technique
        func: Enhancement function
    """
    ENHANCEMENT_FN_MAP[name] = func
    logger.info(f"Registered enhancement function: {name}")


def get_available_enhancements() -> list:
    """Get list of available enhancement techniques."""
    return list(ENHANCEMENT_FN_MAP.keys())

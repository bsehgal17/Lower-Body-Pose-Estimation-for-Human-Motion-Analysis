"""
Gamma Correction Enhancement Module

This module provides functionality to apply gamma correction to video frames for improved
brightness and contrast, particularly useful for correcting exposure issues and enhancing
visibility in various lighting conditions.
"""

import cv2
import numpy as np
import logging
from typing import Tuple, Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)


class GammaEnhancer:
    """
    A class for applying gamma correction to images and videos.

    Gamma correction is a power-law transformation that adjusts the luminance or
    tristimulus values of an image. It can brighten or darken images in a non-linear
    way, preserving details better than simple brightness adjustment.
    """

    def __init__(self, gamma: float):
        """
        Initialize the gamma enhancer.

        Args:
            gamma (float): Gamma correction value.
                          Values < 1.0 make image brighter (good for dark images)
                          Values > 1.0 make image darker (good for overexposed images)
                          Value = 1.0 leaves image unchanged
                          Typical range: 0.5-2.5
        """
        self.gamma = gamma

        # Pre-compute the lookup table for efficiency
        self._build_lookup_table()

        logger.info(f"Gamma enhancer initialized with gamma={gamma}")

    def _build_lookup_table(self):
        """Build a lookup table for gamma correction to improve performance."""
        # Create a lookup table mapping each pixel value [0, 255] to its gamma-corrected value
        inv_gamma = 1.0 / self.gamma
        self.lookup_table = np.array(
            [((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]
        ).astype("uint8")

    def enhance_frame(self, frame: np.ndarray, color_space: str = "HSV") -> np.ndarray:
        """
        Apply gamma correction to a single frame.

        Args:
            frame (np.ndarray): Input frame in BGR color space
            color_space (str): Color space for gamma application ('BGR', 'LAB', 'HSV', 'YUV', 'GRAY')
                              Default: 'HSV' (recommended for most cases)

        Returns:
            np.ndarray: Gamma-corrected frame in BGR color space
        """
        if frame is None or frame.size == 0:
            logger.warning("Empty or None frame provided to enhance_frame")
            return frame

        # Handle grayscale images
        if len(frame.shape) == 2:
            return cv2.LUT(frame, self.lookup_table)

        # Handle color images based on color space
        if color_space.upper() == "HSV":
            # Convert to HSV and apply gamma to V channel
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            h_channel, s_channel, v_channel = cv2.split(hsv)

            # Apply gamma correction to V channel (value/brightness)
            v_channel_gamma = cv2.LUT(v_channel, self.lookup_table)

            # Merge channels back
            hsv_gamma = cv2.merge((h_channel, s_channel, v_channel_gamma))
            return cv2.cvtColor(hsv_gamma, cv2.COLOR_HSV2BGR)

        elif color_space.upper() == "BGR":
            # Apply gamma correction to all channels
            return cv2.LUT(frame, self.lookup_table)

        elif color_space.upper() == "LAB":
            # Convert to LAB and apply gamma to L channel only
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)

            # Apply gamma correction to L channel (luminance)
            l_channel_gamma = cv2.LUT(l_channel, self.lookup_table)

            # Merge channels back
            lab_gamma = cv2.merge((l_channel_gamma, a_channel, b_channel))
            return cv2.cvtColor(lab_gamma, cv2.COLOR_LAB2BGR)

        elif color_space.upper() == "YUV":
            # Convert to YUV and apply gamma to Y channel
            yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
            y_channel, u_channel, v_channel = cv2.split(yuv)

            # Apply gamma correction to Y channel (luminance)
            y_channel_gamma = cv2.LUT(y_channel, self.lookup_table)

            # Merge channels back
            yuv_gamma = cv2.merge((y_channel_gamma, u_channel, v_channel))
            return cv2.cvtColor(yuv_gamma, cv2.COLOR_YUV2BGR)

        elif color_space.upper() == "GRAY":
            # Convert to grayscale, apply gamma, then back to BGR
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gamma_gray = cv2.LUT(gray, self.lookup_table)
            return cv2.cvtColor(gamma_gray, cv2.COLOR_GRAY2BGR)

        else:
            logger.warning(f"Unsupported color space: {color_space}. Using HSV.")
            return cv2.LUT(frame, self.lookup_table)

    def enhance_video(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        color_space: str,
        progress_callback: Optional[callable] = None,
    ) -> bool:
        """
        Apply gamma correction to an entire video file.

        Args:
            input_path (Union[str, Path]): Path to input video file
            output_path (Union[str, Path]): Path to output enhanced video file
            color_space (str): Color space for gamma application
            progress_callback (Optional[callable]): Callback function for progress updates

        Returns:
            bool: True if successful, False otherwise
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        if not input_path.exists():
            logger.error(f"Input video file not found: {input_path}")
            return False

        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Open input video
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            logger.error(f"Could not open video: {input_path}")
            return False

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        if not out.isOpened():
            logger.error(f"Could not create output video: {output_path}")
            cap.release()
            return False

        logger.info(f"Processing video: {input_path}")
        logger.info(
            f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames"
        )

        frame_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Apply gamma correction
                enhanced_frame = self.enhance_frame(frame, color_space)

                # Write enhanced frame
                out.write(enhanced_frame)

                frame_count += 1

                # Progress callback
                if (
                    progress_callback and frame_count % 30 == 0
                ):  # Update every 30 frames
                    progress = (frame_count / total_frames) * 100
                    progress_callback(frame_count, total_frames, progress)

        except Exception as e:
            logger.error(f"Error processing video: {e}")
            return False

        finally:
            cap.release()
            out.release()

        logger.info(f"Video enhancement completed: {output_path}")
        logger.info(f"Processed {frame_count} frames")

        return True

    def update_parameters(self, gamma: Optional[float] = None):
        """
        Update gamma correction parameters and recreate the lookup table.

        Args:
            gamma (Optional[float]): New gamma value
        """
        if gamma is not None:
            self.gamma = gamma
            self._build_lookup_table()

        logger.info(f"Gamma parameters updated: gamma={self.gamma}")


def apply_gamma_to_frame(
    frame: np.ndarray,
    gamma: float = 1.0,
    color_space: str = "HSV",
) -> np.ndarray:
    """
    Convenience function to apply gamma correction to a single frame.

    Args:
        frame (np.ndarray): Input frame in BGR color space
        gamma (float): Gamma correction value
        color_space (str): Color space for gamma application

    Returns:
        np.ndarray: Gamma-corrected frame in BGR color space
    """
    enhancer = GammaEnhancer(gamma=gamma)
    return enhancer.enhance_frame(frame, color_space)


def batch_enhance_videos(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    gamma: float = 1.0,
    color_space: str = "HSV",
    file_extensions: Tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv"),
    progress_callback: Optional[callable] = None,
) -> bool:
    """
    Apply gamma correction to all videos in a directory.

    Args:
        input_dir (Union[str, Path]): Directory containing input videos
        output_dir (Union[str, Path]): Directory for output videos
        gamma (float): Gamma correction value
        color_space (str): Color space for gamma application
        file_extensions (Tuple[str, ...]): Video file extensions to process
        progress_callback (Optional[callable]): Progress callback function

    Returns:
        bool: True if all videos processed successfully, False otherwise
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return False

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all video files
    video_files = []
    for ext in file_extensions:
        video_files.extend(input_dir.glob(f"*{ext}"))

    if not video_files:
        logger.warning(f"No video files found in: {input_dir}")
        return True

    logger.info(f"Found {len(video_files)} video files to process")

    # Create enhancer
    enhancer = GammaEnhancer(gamma=gamma)

    success_count = 0
    for i, video_file in enumerate(video_files):
        output_file = output_dir / f"gamma_{video_file.name}"

        logger.info(f"Processing {i + 1}/{len(video_files)}: {video_file.name}")

        # Create progress callback for this video
        def video_progress(frame_num, total_frames, progress):
            if progress_callback:
                overall_progress = ((i * 100) + progress) / len(video_files)
                progress_callback(
                    f"Video {i + 1}/{len(video_files)}: {progress:.1f}%",
                    overall_progress,
                )

        success = enhancer.enhance_video(
            input_path=video_file,
            output_path=output_file,
            color_space=color_space,
            progress_callback=video_progress,
        )

        if success:
            success_count += 1
            logger.info(f"Successfully processed: {video_file.name}")
        else:
            logger.error(f"Failed to process: {video_file.name}")

    logger.info(
        f"Batch processing completed: {success_count}/{len(video_files)} videos processed"
    )
    return success_count == len(video_files)

"""
CLAHE (Contrast Limited Adaptive Histogram Equalization) Enhancement Module

This module provides functionality to apply CLAHE to video frames for improved contrast
and visibility, particularly useful for low-light or poor contrast conditions.
"""

import cv2
import numpy as np
import logging
from typing import Tuple, Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)


class CLAHEEnhancer:
    """
    A class for applying CLAHE (Contrast Limited Adaptive Histogram Equalization) to images and videos.

    CLAHE is an adaptive histogram equalization technique that enhances local contrast
    while limiting noise amplification in homogeneous areas.
    """

    def __init__(self, clip_limit: float, tile_grid_size: Tuple[int, int]):
        """
        Initialize the CLAHE enhancer.

        Args:
            clip_limit (float): Threshold for contrast limiting. Higher values allow more contrast.
                               Typical range: 1.0-4.0. Default: 2.0
            tile_grid_size (Tuple[int, int]): Size of the grid for local histogram equalization.
                                            Each tile is processed independently. Default: (8, 8)
        """
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

        logger.info(
            f"CLAHE enhancer initialized with clip_limit={clip_limit}, "
            f"tile_grid_size={tile_grid_size}"
        )

    def enhance_frame(self, frame: np.ndarray, color_space: str = "LAB") -> np.ndarray:
        """
        Apply CLAHE enhancement to a single frame.

        Args:
            frame (np.ndarray): Input frame in BGR color space
            color_space (str): Color space for CLAHE application ('LAB', 'HSV', 'YUV', 'GRAY')
                              Default: 'LAB' (recommended for most cases)

        Returns:
            np.ndarray: Enhanced frame in BGR color space
        """
        if frame is None or frame.size == 0:
            logger.warning("Empty or None frame provided to enhance_frame")
            return frame

        # Handle grayscale images
        if len(frame.shape) == 2:
            return self.clahe.apply(frame)

        # Handle color images
        if color_space.upper() == "LAB":
            # Convert to LAB and apply CLAHE to L channel
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)

            # Apply CLAHE to L channel (luminance)
            l_channel_clahe = self.clahe.apply(l_channel)

            # Merge channels back
            lab_clahe = cv2.merge((l_channel_clahe, a_channel, b_channel))
            return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

        elif color_space.upper() == "HSV":
            # Convert to HSV and apply CLAHE to V channel
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            h_channel, s_channel, v_channel = cv2.split(hsv)

            # Apply CLAHE to V channel (value/brightness)
            v_channel_clahe = self.clahe.apply(v_channel)

            # Merge channels back
            hsv_clahe = cv2.merge((h_channel, s_channel, v_channel_clahe))
            return cv2.cvtColor(hsv_clahe, cv2.COLOR_HSV2BGR)

        elif color_space.upper() == "YUV":
            # Convert to YUV and apply CLAHE to Y channel
            yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
            y_channel, u_channel, v_channel = cv2.split(yuv)

            # Apply CLAHE to Y channel (luminance)
            y_channel_clahe = self.clahe.apply(y_channel)

            # Merge channels back
            yuv_clahe = cv2.merge((y_channel_clahe, u_channel, v_channel))
            return cv2.cvtColor(yuv_clahe, cv2.COLOR_YUV2BGR)

        elif color_space.upper() == "GRAY":
            # Convert to grayscale, apply CLAHE, then back to BGR
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_clahe = self.clahe.apply(gray)
            return cv2.cvtColor(gray_clahe, cv2.COLOR_GRAY2BGR)

        else:
            logger.error(f"Unsupported color space: {color_space}")
            return frame

    def enhance_video(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        color_space: str,
        progress_callback: Optional[callable] = None,
    ) -> bool:
        """
        Apply CLAHE enhancement to an entire video file.

        Args:
            input_path (Union[str, Path]): Path to input video file
            output_path (Union[str, Path]): Path to output enhanced video file
            color_space (str): Color space for CLAHE application
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
            logger.error(f"Could not open input video: {input_path}")
            return False

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Define codec and create VideoWriter
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

                # Apply CLAHE enhancement
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

    def update_parameters(
        self,
        clip_limit: Optional[float] = None,
        tile_grid_size: Optional[Tuple[int, int]] = None,
    ):
        """
        Update CLAHE parameters and recreate the CLAHE object.

        Args:
            clip_limit (Optional[float]): New clip limit value
            tile_grid_size (Optional[Tuple[int, int]]): New tile grid size
        """
        if clip_limit is not None:
            self.clip_limit = clip_limit

        if tile_grid_size is not None:
            self.tile_grid_size = tile_grid_size

        # Recreate CLAHE object with new parameters
        self.clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size
        )

        logger.info(
            f"CLAHE parameters updated: clip_limit={self.clip_limit}, "
            f"tile_grid_size={self.tile_grid_size}"
        )


def apply_clahe_to_frame(
    frame: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
    color_space: str = "LAB",
) -> np.ndarray:
    """
    Convenience function to apply CLAHE to a single frame.

    Args:
        frame (np.ndarray): Input frame in BGR color space
        clip_limit (float): Threshold for contrast limiting
        tile_grid_size (Tuple[int, int]): Size of the grid for local histogram equalization
        color_space (str): Color space for CLAHE application

    Returns:
        np.ndarray: Enhanced frame in BGR color space
    """
    enhancer = CLAHEEnhancer(clip_limit=clip_limit, tile_grid_size=tile_grid_size)
    return enhancer.enhance_frame(frame, color_space)


def batch_enhance_videos(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
    color_space: str = "LAB",
    file_extensions: Tuple[str, ...] = (".mp4", ".avi", ".mov"),
):
    """
    Apply CLAHE enhancement to all videos in a directory.

    Args:
        input_dir (Union[str, Path]): Directory containing input videos
        output_dir (Union[str, Path]): Directory for enhanced videos
        clip_limit (float): Threshold for contrast limiting
        tile_grid_size (Tuple[int, int]): Size of the grid for local histogram equalization
        color_space (str): Color space for CLAHE application
        file_extensions (Tuple[str, ...]): Video file extensions to process
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize CLAHE enhancer
    enhancer = CLAHEEnhancer(clip_limit=clip_limit, tile_grid_size=tile_grid_size)

    # Find all video files
    video_files = []
    for ext in file_extensions:
        video_files.extend(input_dir.glob(f"**/*{ext}"))

    if not video_files:
        logger.warning(f"No video files found in {input_dir}")
        return

    logger.info(f"Found {len(video_files)} video files to process")

    # Process each video
    for i, video_path in enumerate(video_files, 1):
        # Create relative output path
        relative_path = video_path.relative_to(input_dir)
        output_path = output_dir / f"clahe_{relative_path.stem}{relative_path.suffix}"

        logger.info(f"Processing {i}/{len(video_files)}: {video_path.name}")

        def progress_callback(frame_num, total_frames, progress):
            print(
                f"\r  Progress: {progress:.1f}% ({frame_num}/{total_frames} frames)",
                end="",
            )

        success = enhancer.enhance_video(
            video_path, output_path, color_space, progress_callback
        )

        if success:
            print(f"\n  ✓ Completed: {output_path.name}")
        else:
            print(f"\n  ✗ Failed: {video_path.name}")

    logger.info("Batch enhancement completed")

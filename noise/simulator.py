import os
import cv2
import logging
from typing import Optional
from config.base import GlobalConfig
from utils.utils import get_video_files
from .types import add_realistic_noise, apply_motion_blur

logger = logging.getLogger(__name__)


class NoiseSimulator:
    """
    Class to simulate realistic noise on videos by applying Poisson, Gaussian, and motion blur.
    """

    def __init__(self, config: GlobalConfig):
        self.config = config
        self.noise_params = config.noise

    def _apply_brightness_reduction(self, frame):
        """
        Reduces brightness of the frame by subtracting from the V channel in HSV space.
        """
        if not self.noise_params.apply_brightness_reduction:
            return frame

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.subtract(v, self.noise_params.brightness_factor)
        hsv_modified = cv2.merge((h, s, v))
        return cv2.cvtColor(hsv_modified, cv2.COLOR_HSV2BGR)

    def process_single_video(self, input_path: str, output_path: str):
        """
        Processes a single video file and writes a noisy version to output.

        Args:
            input_path (str): Path to original video.
            output_path (str): Path where noisy video will be saved.
        """
        logger.info(f"Processing video for noise: {input_path} -> {output_path}")
        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():
            logger.error(f"Failed to open video: {input_path}. Skipping.")
            return

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        target_res = self.noise_params.target_resolution or (orig_width, orig_height)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, target_res)

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame.shape[:2] != target_res[::-1]:  # (W, H) to (H, W)
                frame = cv2.resize(frame, target_res)

            frame = self._apply_brightness_reduction(frame)
            frame = add_realistic_noise(
                frame,
                poisson_scale=self.noise_params.poisson_scale,
                gaussian_std=self.noise_params.gaussian_std,
            )
            frame = apply_motion_blur(
                frame, kernel_size=self.noise_params.motion_blur_kernel_size
            )

            out.write(frame)
            frame_idx += 1
            logger.debug(f"Processed frame {frame_idx}")

        cap.release()
        out.release()
        logger.info(f"Finished writing noisy video to {output_path}")

    def process_all_videos(self, input_folder: str, output_folder: str):
        """
        Processes all video files in the input folder and writes noisy versions.

        Args:
            input_folder (str): Folder containing input videos.
            output_folder (str): Folder to write output videos (same structure).
        """
        logger.info(f"Starting noise simulation for videos in: {input_folder}")
        video_files = get_video_files(input_folder, self.config.video.extensions)

        if not video_files:
            logger.warning("No video files found to process.")
            return

        for input_path in video_files:
            relative_path = os.path.relpath(input_path, input_folder)
            output_path = os.path.join(output_folder, relative_path)
            self.process_single_video(input_path, output_path)

        logger.info("Completed noise simulation for all videos.")

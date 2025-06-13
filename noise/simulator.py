import os
import cv2
import logging
from typing import Optional
from config.base import GlobalConfig
from utils.video_io import get_video_files
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
        if not getattr(self.noise_params, "apply_brightness_reduction", False):
            return frame

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.subtract(v, self.noise_params.brightness_factor)
        hsv_modified = cv2.merge((h, s, v))
        return cv2.cvtColor(hsv_modified, cv2.COLOR_HSV2BGR)

    def _apply_noise(self, frame):
        if getattr(self.noise_params, "apply_poisson_noise", True) or getattr(self.noise_params, "apply_gaussian_noise", True):
            frame = add_realistic_noise(
                frame,
                poisson_scale=self.noise_params.poisson_scale if getattr(
                    self.noise_params, "apply_poisson_noise", True) else 0,
                gaussian_std=self.noise_params.gaussian_std if getattr(
                    self.noise_params, "apply_gaussian_noise", True) else 0,
            )
        return frame

    def _apply_motion_blur(self, frame):
        if getattr(self.noise_params, "apply_motion_blur", True):
            return apply_motion_blur(
                frame, kernel_size=self.noise_params.motion_blur_kernel_size
            )
        return frame

    def process_single_video(self, input_path: str, output_path: str):
        logger.info(
            f"Processing video for noise: {input_path} -> {output_path}")
        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():
            logger.error(f"Failed to open video: {input_path}. Skipping.")
            return

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Default to original resolution if not specified
        target_res = self.noise_params.target_resolution or (
            orig_width, orig_height)

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
            frame = self._apply_noise(frame)
            frame = self._apply_motion_blur(frame)

            out.write(frame)
            frame_idx += 1
            logger.debug(f"Processed frame {frame_idx}")

        cap.release()
        out.release()
        logger.info(f"Finished writing noisy video to {output_path}")

    def process_all_videos(self, input_folder: str, output_folder: str):
        logger.info(f"Starting noise simulation for videos in: {input_folder}")
        video_files = get_video_files(
            input_folder, self.config.video.extensions)

        if not video_files:
            logger.warning("No video files found to process.")
            return

        for input_path in video_files:
            relative_path = os.path.relpath(input_path, input_folder)
            output_path = os.path.join(output_folder, relative_path)
            self.process_single_video(input_path, output_path)

        logger.info("Completed noise simulation for all videos.")

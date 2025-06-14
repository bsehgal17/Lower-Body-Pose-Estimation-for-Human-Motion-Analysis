import os
import cv2
import logging
from config.pipeline_config import PipelineConfig
from config.global_config import GlobalConfig
from utils.video_io import get_video_files
from noise.brightness import apply_brightness_reduction
from noise.real_world_noise import apply_combined_noise
from noise.motion_blur import apply_motion_blur

logger = logging.getLogger(__name__)


class NoiseSimulator:
    def __init__(self, pipeline_config: PipelineConfig, global_config: GlobalConfig):
        self.pipeline_config = pipeline_config
        self.global_config = global_config
        self.noise_params = pipeline_config.noise
        self.video_exts = global_config.video.extensions

    def _apply_pipeline(self, frame):
        frame = apply_brightness_reduction(frame, self.noise_params)
        frame = apply_combined_noise(frame, self.noise_params)
        frame = apply_motion_blur(frame, self.noise_params)
        return frame

    def process_single_video(self, input_path: str, output_path: str):
        logger.info(f"Processing video: {input_path} -> {output_path}")
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video: {input_path}")
            return

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        target_res = self.noise_params.target_resolution or (width, height)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        out = cv2.VideoWriter(
            output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, target_res
        )

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame.shape[:2] != target_res[::-1]:
                frame = cv2.resize(frame, target_res)
            out.write(self._apply_pipeline(frame))
            frame_idx += 1
            logger.debug(f"Processed frame {frame_idx}")

        cap.release()
        out.release()
        logger.info(f"Finished writing to {output_path}")

    def process_all_videos(self, input_folder: str, output_folder: str):
        video_files = get_video_files(input_folder, self.video_exts)
        if not video_files:
            logger.warning("No video files found.")
            return

        for input_path in video_files:
            rel_path = os.path.relpath(input_path, input_folder)
            output_path = os.path.join(output_folder, rel_path)
            self.process_single_video(input_path, output_path)

        logger.info("All videos processed with noise simulation.")

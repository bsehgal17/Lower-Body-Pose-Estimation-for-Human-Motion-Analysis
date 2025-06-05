import os
import logging
from config.base import GlobalConfig
from utils.video_io import get_video_files
from utils.json_io import save_keypoints_to_json
from pose_estimation.detector import Detector
from pose_estimation.estimator import PoseEstimator
from pose_estimation.visualizer import PoseVisualizer
from pose_estimation.processors.frame_processor import FrameProcessor
from pose_estimation.processors.video_loader import VideoIO

logger = logging.getLogger(__name__)


def run_detection_pipeline(config: GlobalConfig):
    logger.info("Initializing models...")
    detector = Detector(config)
    estimator = PoseEstimator(config)
    visualizer = PoseVisualizer(estimator, config)
    processor = FrameProcessor(detector, estimator, visualizer, config)

    video_files = get_video_files(config.paths.video_folder, config.video.extensions)
    if not video_files:
        logger.warning(f"No video files found in {config.paths.video_folder}")
        return

    for video_path in video_files:
        video_data = []
        rel_path = os.path.relpath(video_path, config.paths.video_folder)
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        current_save_dir = os.path.join(
            config.paths.output_dir, os.path.dirname(rel_path), video_name
        )
        os.makedirs(current_save_dir, exist_ok=True)

        output_json_file = os.path.join(current_save_dir, f"{video_name}.json")
        output_video_file = os.path.join(current_save_dir, os.path.basename(video_path))

        logger.info(f"Processing {video_path} -> {output_video_file}")

        video_io = VideoIO(video_path, output_video_file)
        frame_idx = 0
        while True:
            ret, frame = video_io.read()
            if not ret:
                break
            processed_frame = processor.process_frame(frame, frame_idx, video_data)
            video_io.write(processed_frame)
            frame_idx += 1

        video_io.release()
        save_keypoints_to_json(video_data, current_save_dir, video_name)
        logger.info(f"Keypoints saved to {output_json_file}")
        logger.info(f"Output video saved to {output_video_file}")

    logger.info("All video processing complete.")

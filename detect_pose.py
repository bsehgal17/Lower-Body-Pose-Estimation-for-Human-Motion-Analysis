import os
import logging
import pickle
from config.pipeline_config import PipelineConfig
from config.global_config import GlobalConfig
from utils.video_io import get_video_files
from utils.json_io import save_keypoints_to_json
from pose_estimation.detector import Detector
from pose_estimation.estimator import PoseEstimator
from pose_estimation.visualization import PoseVisualizer
from pose_estimation.processors.frame_processor import FrameProcessor
from pose_estimation.processors.video_loader import VideoIO


logger = logging.getLogger(__name__)


def run_detection_pipeline(pipeline_config: PipelineConfig, global_config: GlobalConfig, input_dir: str, output_dir: str):
    logger.info("Initializing models...")

    # Initialize components with pipeline_config where applicable
    detector = Detector(pipeline_config)
    estimator = PoseEstimator(pipeline_config)
    visualizer = PoseVisualizer(estimator, pipeline_config)
    processor = FrameProcessor(
        detector, estimator, visualizer, pipeline_config)

    video_files = get_video_files(input_dir, global_config.video.extensions)
    if not video_files:
        logger.warning(f"No video files found in {input_dir}")
        return

    for video_path in video_files:
        video_data = []

        rel_path = os.path.relpath(video_path, input_dir)
        video_name = os.path.splitext(os.path.basename(video_path))[0]

        current_save_dir = os.path.join(
            output_dir, os.path.dirname(rel_path), video_name)
        os.makedirs(current_save_dir, exist_ok=True)

        output_json_file = os.path.join(current_save_dir, f"{video_name}.json")
        output_pkl_file = os.path.join(current_save_dir, f"{video_name}.pkl")
        output_video_file = os.path.join(
            current_save_dir, os.path.basename(video_path))

        logger.info(f"Processing {video_path} -> {output_video_file}")

        video_io = VideoIO(video_path, output_video_file)
        frame_idx = 0

        while True:
            ret, frame = video_io.read()
            if not ret:
                break

            processed_frame = processor.process_frame(
                frame, frame_idx, video_data)
            video_io.write(processed_frame)
            frame_idx += 1

        video_io.release()

        # Save results
        save_keypoints_to_json(video_data, current_save_dir, video_name)
        with open(output_pkl_file, "wb") as f:
            pickle.dump(video_data, f)

        logger.info(
            f"Keypoints saved to {output_json_file} and {output_pkl_file}")
        logger.info(f"Output video saved to {output_video_file}")

    logger.info("All video processing complete.")

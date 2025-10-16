import os
import logging

# asdict no longer needed - using Pydantic models
from config.pipeline_config import PipelineConfig
from config.global_config import GlobalConfig
from utils.video_io import get_video_files
from utils.standard_saver import save_standard_format, SavedData
from utils.data_structures import VideoData
from pose_estimation.rtmw.detector_rtmw import Detector
from pose_estimation.rtmw.estimator_rtmw import PoseEstimator
from pose_estimation.rtmw.visualization import PoseVisualizer
from pose_estimation.processors.frame_processor import FrameProcessor
from pose_estimation.processors.video_loader import VideoIO


logger = logging.getLogger(__name__)


def run_detection_pipeline(
    pipeline_config: PipelineConfig,
    global_config: GlobalConfig,
    input_dir: str,
    output_dir: str,
):
    logger.info("Initializing models...")

    # Initialize components with pipeline_config where applicable
    detector = Detector(pipeline_config)
    estimator = PoseEstimator(pipeline_config)
    visualizer = PoseVisualizer(estimator, pipeline_config)
    processor = FrameProcessor(detector, estimator, visualizer, pipeline_config)

    video_files = get_video_files(input_dir, global_config.video.extensions)
    if not video_files:
        logger.warning(f"No video files found in {input_dir}")
        return

    for video_path in video_files:
        # Create VideoData object for this video
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        video_data = VideoData(video_name=video_name)

        rel_path = os.path.relpath(video_path, input_dir)

        current_save_dir = os.path.join(
            output_dir, os.path.dirname(rel_path), video_name
        )
        os.makedirs(current_save_dir, exist_ok=True)

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

        # Finalize person tracking across all frames
        processor.finalize_video_processing(video_data)

        # Set detection config
        detector_config_dict = pipeline_config.processing.model_dump()
        video_data.detection_config = detector_config_dict

        # Create SavedData with detection configuration
        saved_data = SavedData.from_video_data(
            video_data,
            detection_config=detector_config_dict,
            processing_metadata={
                "pipeline": "rtmw",
                "input_video": video_path,
                "output_dir": current_save_dir,
            },
        )

        # Save using standard_saver
        save_standard_format(
            data=saved_data,
            output_dir=current_save_dir,
            original_file_path=video_path,
            suffix="",
            save_json=True,
            save_pickle=True,
            save_video_overlay=False,
        )

        logger.info(
            f"Complete video data saved to {current_save_dir}/{video_name}.json and {current_save_dir}/{video_name}.pkl"
        )
        logger.info(f"Output video saved to {output_video_file}")

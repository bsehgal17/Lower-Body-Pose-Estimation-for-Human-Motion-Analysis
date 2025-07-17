import os
import logging
import pickle
import json
from dataclasses import asdict
from config.pipeline_config import PipelineConfig
from config.global_config import GlobalConfig
from utils.video_io import get_video_files
from utils.json_io import save_keypoints_to_json
from pose_estimation.deeplabcut.detector_deeplabcut import DeepLabCutDetector
from pose_estimation.deeplabcut.frame_processor_deeplabcut import FrameProcessorDLC
from pose_estimation.deeplabcut.deeplabcut_visualization import DeepLabCutVisualizer
from pose_estimation.processors.video_loader import VideoIO
from deeplabcut.utils.make_labeled_video import CreateVideo
from deeplabcut.utils.video_processor import VideoProcessorCV
from pose_estimation.deeplabcut.skeleton_config import bodyparts2connect
import deeplabcut.torch as dlc_torch

logger = logging.getLogger(__name__)

def run_detection_pipeline(pipeline_config: PipelineConfig, global_config: GlobalConfig, input_dir: str, output_dir: str):
    logger.info("Initializing models...")

    detector = DeepLabCutDetector(pipeline_config)
    visualizer = DeepLabCutVisualizer()
    processor = FrameProcessorDLC(detector, visualizer, pipeline_config)

    video_files = get_video_files(input_dir, global_config.video.extensions)
    if not video_files:
        logger.warning(f"No video files found in {input_dir}")
        return

    for video_path in video_files:
        video_data = []
        rel_path = os.path.relpath(video_path, input_dir)
        video_name = os.path.splitext(os.path.basename(video_path))[0]

        current_save_dir = os.path.join(output_dir, os.path.dirname(rel_path), video_name)
        os.makedirs(current_save_dir, exist_ok=True)

        output_json_file = os.path.join(current_save_dir, f"{video_name}.json")
        output_pkl_file = os.path.join(current_save_dir, f"{video_name}.pkl")
        output_video_file = os.path.join(current_save_dir, os.path.basename(video_path))
        overlay_video_file = os.path.join(current_save_dir, f"{video_name}_overlay.mp4")

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

        detector_config_dict = asdict(pipeline_config.processing)

        save_keypoints_to_json(
            video_data,
            current_save_dir,
            video_name,
            detector_config=detector_config_dict
        )

        bundle = {
            "keypoints": video_data,
            "detection_config": detector_config_dict
        }

        with open(output_pkl_file, "wb") as f:
            pickle.dump(bundle, f)

        logger.info(f"Keypoints saved to {output_json_file} and {output_pkl_file}")
        logger.info(f"Output video saved to {output_video_file}")

        # --- Create overlay video with DLC visualizer ---
        try:
            pose_cfg = detector.pose_cfg
            df = dlc_torch.build_predictions_dataframe(
                scorer="deeplabcut-body7",
                predictions={idx: frame["bodyparts"] for idx, frame in enumerate(video_data)},
                parameters=dlc_torch.PoseDatasetParameters(
                    bodyparts=pose_cfg["metadata"]["bodyparts"],
                    unique_bpts=pose_cfg["metadata"]["unique_bodyparts"],
                    individuals=[f"idv_{i}" for i in range(len(video_data[0]["bodyparts"]))],
                )
            )

            clip = VideoProcessorCV(str(video_path), sname=overlay_video_file, codec="mp4v")

            CreateVideo(
                clip,
                df,
                pcutoff=0.4,
                dotsize=3,
                colormap="rainbow",
                bodyparts2plot=pose_cfg["metadata"]["bodyparts"],
                trailpoints=0,
                cropping=False,
                x1=0, x2=clip.w,
                y1=0, y2=clip.h,
                bodyparts2connect=bodyparts2connect,
                skeleton_color="w",
                draw_skeleton=True,
                displaycropped=True,
                color_by="bodypart"
            )

            logger.info(f"Overlay video saved to {overlay_video_file}")

        except Exception as e:
            logger.exception(f"Failed to create overlay video: {e}")

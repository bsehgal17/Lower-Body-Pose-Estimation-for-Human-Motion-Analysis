# main_handlers.py
import logging
from config.base import GlobalConfig
from pipeline import run_detection_pipeline
from noise.simulator import NoiseSimulator
from filter_runner import run_keypoint_filtering
from assessment_runner import run_pose_assessment_pipeline

logger = logging.getLogger(__name__)


def _handle_detect_command(args, config: GlobalConfig):
    if args.video_folder:
        config.paths.video_folder = args.video_folder
        logger.info(f"Overriding video folder: {args.video_folder}")
    if args.output_dir:
        config.paths.output_dir = args.output_dir
        logger.info(f"Overriding output dir: {args.output_dir}")
    run_detection_pipeline(config)


def _handle_noise_command(args, config: GlobalConfig):
    input_folder = args.input_folder or config.paths.video_folder
    output_folder = args.output_folder or config.paths.output_dir
    simulator = NoiseSimulator(config)
    simulator.process_all_videos(input_folder, output_folder)


def _handle_assess_command(args, config: GlobalConfig):
    run_pose_assessment_pipeline(config)


def _handle_filter_command(args, config: GlobalConfig):
    params = {}
    for param in args.params:
        if "=" not in param:
            logger.warning(f"Invalid param: {param}")
            continue
        key, value = param.split("=", 1)
        try:
            params[key] = eval(
                value
            )  # Use eval only if trusted; else use type conversion
        except Exception:
            params[key] = value
    run_keypoint_filtering(config, args.filter_name, params)

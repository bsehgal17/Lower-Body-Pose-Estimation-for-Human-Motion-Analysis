import logging
from config.base import GlobalConfig

logger = logging.getLogger(__name__)


def _handle_detect_command(args, config: GlobalConfig):
    from detect_pose import run_detection_pipeline

    if args.video_folder:
        config.paths.video_folder = args.video_folder
        logger.info(f"Overriding video folder: {args.video_folder}")
    if args.output_dir:
        config.paths.output_dir = args.output_dir
        logger.info(f"Overriding output dir: {args.output_dir}")
    run_detection_pipeline(config)


def _handle_noise_command(args, config: GlobalConfig):
    from noise.simulator import NoiseSimulator

    input_folder = args.input_folder or config.paths.video_folder
    output_folder = args.output_folder or config.paths.output_dir
    simulator = NoiseSimulator(config)
    simulator.process_all_videos(input_folder, output_folder)


def _handle_assess_command(args, config: GlobalConfig):
    from assessment_runner import run_pose_assessment_pipeline

    run_pose_assessment_pipeline(config)


def _handle_filter_command(args, config: GlobalConfig):
    from filter_runner import run_keypoint_filtering_from_config
    run_keypoint_filtering_from_config(config)

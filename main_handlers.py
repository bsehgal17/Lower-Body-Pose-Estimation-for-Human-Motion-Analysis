import logging
from pathlib import Path
from config.base import GlobalConfig
from utils.run_utils import make_run_dir  # helper for making run directory

logger = logging.getLogger(__name__)


def _handle_detect_command(args, config: GlobalConfig):
    from detect_pose import run_detection_pipeline

    run_dir = make_run_dir(
        base_out=config.paths.output_dir, pipeline_name="detect", cfg_path=args.config
    )
    config.paths.output_dir = str(run_dir)

    if args.video_folder:
        config.paths.video_folder = args.video_folder
        logger.info(f"Overriding video folder: {args.video_folder}")
    run_detection_pipeline(config)


def _handle_noise_command(args, config: GlobalConfig):
    from noise.simulator import NoiseSimulator

    run_dir = make_run_dir(
        base_out=config.paths.output_dir, pipeline_name="noise", cfg_path=args.config
    )
    config.paths.output_dir = str(run_dir)

    input_folder = args.input_folder or config.paths.video_folder
    output_folder = args.output_folder or config.paths.output_dir

    step_out = Path(config.paths.output_dir) / "noise"
    config.paths.output_dir = str(step_out)

    simulator = NoiseSimulator(config)
    simulator.process_all_videos(input_folder, output_folder)


def _handle_filter_command(args, config: GlobalConfig):
    from filter_runner import run_keypoint_filtering_from_config

    run_dir = make_run_dir(
        base_out=config.paths.output_dir, pipeline_name="filter", cfg_path=args.config
    )
    config.paths.output_dir = str(run_dir)

    run_keypoint_filtering_from_config(config)


def _handle_assess_command(args, config: GlobalConfig):
    from assessment_runner import run_pose_assessment_pipeline

    run_dir = make_run_dir(
        base_out=config.paths.output_dir, pipeline_name="assess", cfg_path=args.config
    )
    config.paths.output_dir = str(run_dir)

    run_pose_assessment_pipeline(config)

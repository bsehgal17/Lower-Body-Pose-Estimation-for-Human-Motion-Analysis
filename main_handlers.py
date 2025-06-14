import logging
from pathlib import Path
from config.pipeline_config import PipelineConfig
from config.global_config import GlobalConfig
from utils.run_utils import make_run_dir, get_pipeline_io_paths

logger = logging.getLogger(__name__)


def _handle_detect_command(args, pipeline_config: PipelineConfig, global_config: GlobalConfig):
    from detect_pose import run_detection_pipeline

    input_dir, base_pipeline_out = get_pipeline_io_paths(
        global_config.paths, pipeline_config.paths.dataset)
    run_dir = make_run_dir(
        base_out=base_pipeline_out,
        pipeline_name=args.pipeline_name,
        step_name="detect",
        cfg_path=args.pipeline_config,
        global_config_obj=global_config,
        pipeline_config_obj=pipeline_config,
    )

    step_out = run_dir
    step_out.mkdir(parents=True, exist_ok=True)

    video_folder = args.video_folder if args.video_folder else input_dir
    if args.video_folder:
        logger.info(f"Overriding video folder: {video_folder}")

    run_detection_pipeline(
        pipeline_config=pipeline_config,
        global_config=global_config,
        input_dir=video_folder,
        output_dir=step_out
    )


def _handle_noise_command(args, pipeline_config: PipelineConfig, global_config: GlobalConfig):
    from noise.simulator import NoiseSimulator

    input_dir, base_pipeline_out = get_pipeline_io_paths(
        global_config.paths, pipeline_config.paths.dataset)
    run_dir = make_run_dir(
        base_out=base_pipeline_out,
        pipeline_name=args.pipeline_name,
        step_name="noise",
        cfg_path=args.pipeline_config,
        global_config_obj=global_config,
        pipeline_config_obj=pipeline_config,
    )

    step_out = run_dir
    step_out.mkdir(parents=True, exist_ok=True)

    input_folder = args.input_folder or input_dir
    output_folder = args.output_folder or step_out

    simulator = NoiseSimulator(pipeline_config, global_config)
    simulator.process_all_videos(str(input_folder), str(output_folder))


def _handle_filter_command(args, pipeline_config: PipelineConfig, global_config: GlobalConfig):
    from filter_runner import run_keypoint_filtering_from_config

    _, base_pipeline_out = get_pipeline_io_paths(
        global_config.paths, pipeline_config.paths.dataset)
    run_dir = make_run_dir(
        base_out=base_pipeline_out,
        pipeline_name=args.pipeline_name,
        step_name="filter",
        cfg_path=args.pipeline_config,
        global_config_obj=global_config,
        pipeline_config_obj=pipeline_config,
    )

    step_out = run_dir
    step_out.mkdir(parents=True, exist_ok=True)

    run_keypoint_filtering_from_config(
        pipeline_config, global_config, output_dir=step_out)


def _handle_assess_command(args, pipeline_config: PipelineConfig, global_config: GlobalConfig):
    from assessment_runner import run_pose_assessment_pipeline

    _, base_pipeline_out = get_pipeline_io_paths(
        global_config.paths, pipeline_config.paths.dataset)
    run_dir = make_run_dir(
        base_out=base_pipeline_out,
        pipeline_name=args.pipeline_name,
        step_name="assess",
        cfg_path=args.pipeline_config,
        global_config_obj=global_config,
        pipeline_config_obj=pipeline_config,
    )

    step_out = run_dir
    step_out.mkdir(parents=True, exist_ok=True)

    run_pose_assessment_pipeline(
        pipeline_config, global_config, output_dir=step_out)

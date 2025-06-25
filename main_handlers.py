import logging
import os
from config.pipeline_config import PipelineConfig
from config.global_config import GlobalConfig
from utils.run_utils import make_run_dir, get_pipeline_io_paths

logger = logging.getLogger(__name__)


def _handle_detect_command(
    args, pipeline_config: PipelineConfig, global_config: GlobalConfig
):
    from detection_pipeline import run_detection_pipeline

    input_dir, base_pipeline_out = get_pipeline_io_paths(
        global_config.paths, pipeline_config.paths.dataset
    )
    run_dir = make_run_dir(
        base_out=base_pipeline_out,
        pipeline_name=args.pipeline_name,
        step_name=args.command,
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
        output_dir=step_out,
    )


def _handle_noise_command(
    args, pipeline_config: PipelineConfig, global_config: GlobalConfig
):
    from noise_simulator import NoiseSimulator

    input_dir, base_pipeline_out = get_pipeline_io_paths(
        global_config.paths, pipeline_config.paths.dataset
    )
    run_dir = make_run_dir(
        base_out=base_pipeline_out,
        pipeline_name=args.pipeline_name,
        step_name=args.command,
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


def _handle_filter_command(
    args, pipeline_config: PipelineConfig, global_config: GlobalConfig
):
    from filter_pipeline import run_keypoint_filtering_from_config

    input_dir, base_pipeline_out = get_pipeline_io_paths(
        global_config.paths, pipeline_config.paths.dataset
    )

    run_dir = make_run_dir(
        base_out=base_pipeline_out,
        pipeline_name=args.pipeline_name,
        step_name=args.command,
        cfg_path=args.pipeline_config,
        global_config_obj=global_config,
        pipeline_config_obj=pipeline_config,
    )

    step_out = run_dir / args.command
    step_out.mkdir(parents=True, exist_ok=True)

    # Set output_dir as per convention
    pipeline_config.paths.output_dir = str(step_out)

    # Resolve input dir from pipeline_config.filter.input_dir or fallback
    if not pipeline_config.filter.input_dir:
        noise_dir = os.path.join(
            base_pipeline_out, args.pipeline_name, "noise")
        detect_dir = os.path.join(
            base_pipeline_out, args.pipeline_name, "detect")

        if os.path.exists(noise_dir):
            pipeline_config.filter.input_dir = noise_dir
            logger.info(
                f"Auto-selected input_dir from noise step: {noise_dir}")
        elif os.path.exists(detect_dir):
            pipeline_config.filter.input_dir = detect_dir
            logger.info(
                f"Auto-selected input_dir from detect step: {detect_dir}")
        else:
            logger.warning(
                "Could not auto-resolve input_dir for filter. You may need to specify it manually."
            )

    run_keypoint_filtering_from_config(
        pipeline_config, global_config, output_dir=step_out
    )


def _handle_assess_command(
    args, pipeline_config: PipelineConfig, global_config: GlobalConfig
):
    from evaluation_pipeline import run_pose_assessment_pipeline

    input_dir, base_pipeline_out = get_pipeline_io_paths(
        global_config.paths, pipeline_config.paths.dataset
    )
    run_dir = make_run_dir(
        base_out=base_pipeline_out,
        pipeline_name=args.pipeline_name,
        step_name=args.command,
        cfg_path=args.pipeline_config,
        global_config_obj=global_config,
        pipeline_config_obj=pipeline_config,
    )

    step_out = run_dir
    step_out.mkdir(parents=True, exist_ok=True)

    run_pose_assessment_pipeline(
        pipeline_config, global_config, output_dir=step_out, input_dir=input_dir
    )

import logging
import os
from pathlib import Path
from config.pipeline_config import PipelineConfig
from config.global_config import GlobalConfig
from utils.run_utils import make_run_dir, get_pipeline_io_paths

logger = logging.getLogger(__name__)


def _find_enhanced_videos_dir(pipeline_name: str, global_config: GlobalConfig):
    """
    Look for enhanced videos directory created by enhancement pipeline.
    Returns the enhanced videos directory if found, None otherwise.
    """
    base_output_dir = Path(global_config.paths.output_dir)

    # The enhancement pipeline saves videos to: base_output_dir/pipeline_name/enhance/
    enhance_dir = base_output_dir / pipeline_name / "enhance"

    if enhance_dir.exists():
        # Look for video files in the enhancement output directory
        video_extensions = global_config.video.extensions
        video_files = []

        for ext in video_extensions:
            video_files.extend(list(enhance_dir.glob(f"**/*{ext}")))

        if video_files:
            logger.info(f"Found enhanced videos directory: {enhance_dir}")
            return str(enhance_dir)

    # Also check for files with enhancement prefixes in base output directory
    video_extensions = global_config.video.extensions
    for ext in video_extensions:
        enhanced_files = list(base_output_dir.glob(f"**/*clahe*{ext}"))
        enhanced_files.extend(
            list(base_output_dir.glob(f"**/*enhanced*{ext}")))
        enhanced_files.extend(
            list(base_output_dir.glob(f"**/*histogram*{ext}")))
        if enhanced_files:
            enhanced_dir = enhanced_files[0].parent
            logger.info(f"Found enhanced videos by prefix in: {enhanced_dir}")
            return str(enhanced_dir)

    return None


def _find_enhanced_detection_results_dir(
    pipeline_name: str, global_config: GlobalConfig
):
    """
    Look for detection results that were run on enhanced videos.
    Returns the enhanced detection results directory if found, None otherwise.
    """
    base_output_dir = Path(global_config.paths.output_dir)

    # Look for detection results from enhanced pipeline
    # The detection pipeline with enhanced videos creates: base_output_dir/pipeline_name_enhanced/detect/
    enhanced_detection_dir = base_output_dir / \
        f"{pipeline_name}_enhanced" / "detect"

    if enhanced_detection_dir.exists() and any(enhanced_detection_dir.iterdir()):
        logger.info(
            f"Found enhanced detection results directory: {enhanced_detection_dir}"
        )
        return str(enhanced_detection_dir)

    # Also check for detection results that contain enhanced video indicators
    detection_dir = base_output_dir / pipeline_name / "detect"
    if detection_dir.exists():
        for detection_subdir in detection_dir.iterdir():
            if detection_subdir.is_dir():
                # Look for enhanced video indicators in detection folder names or files
                if any(
                    keyword in str(detection_subdir).lower()
                    for keyword in ["clahe", "enhanced", "histogram", "brightness"]
                ):
                    logger.info(
                        f"Found enhanced detection results directory: {detection_dir}"
                    )
                    return str(detection_dir)

                # Or check for JSON files with enhanced video names
                json_files = list(detection_subdir.glob("*.json"))
                for json_file in json_files:
                    if any(
                        keyword in json_file.stem.lower()
                        for keyword in ["clahe", "enhanced", "histogram", "brightness"]
                    ):
                        logger.info(
                            f"Found enhanced detection results directory: {detection_dir}"
                        )
                        return str(detection_dir)

    return None


def _handle_detect_command(
    args, pipeline_config: PipelineConfig, global_config: GlobalConfig
):
    run_detection_pipeline = _get_detection_pipeline_fn(
        pipeline_config.models.detector)

    input_dir, base_pipeline_out = get_pipeline_io_paths(
        global_config.paths, pipeline_config.paths.dataset
    )

    # Check if enhanced videos exist from previous pipeline step
    enhanced_videos_dir = _find_enhanced_videos_dir(
        args.pipeline_name, global_config)

    if enhanced_videos_dir:
        logger.info(
            f"Found enhanced videos, using them for detection: {enhanced_videos_dir}"
        )
        input_dir = enhanced_videos_dir
        # Add enhanced indicator to output directory
        run_dir = make_run_dir(
            base_out=base_pipeline_out,
            pipeline_name=f"{args.pipeline_name}_enhanced",
            step_name=args.command,
            cfg_path=args.pipeline_config,
            global_config_obj=global_config,
            pipeline_config_obj=pipeline_config,
        )
    else:
        logger.info(
            f"No enhanced videos found, using original videos: {input_dir}")
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

    # Step 0: Set up base input/output paths
    input_dir, base_pipeline_out = get_pipeline_io_paths(
        global_config.paths, pipeline_config.paths.dataset
    )

    # Step 1: Add noise to input videos
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

    logger.info(f"Applying noise to videos from: {input_folder}")
    logger.info(f"Noisy videos will be saved to: {output_folder}")

    simulator = NoiseSimulator(pipeline_config, global_config)
    simulator.process_all_videos(str(input_folder), str(output_folder))

    # Step 2: Conditionally run detection only if models and detector are configured
    if (
        pipeline_config.models is not None
        and getattr(pipeline_config.models, "detector", None) is not None
    ):
        run_detection_pipeline = _get_detection_pipeline_fn(
            pipeline_config.models.detector
        )

        detect_out_dir = Path(output_folder) / "detect"
        detect_out_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Running detection on noisy videos from: {output_folder}")
        logger.info(f"Detection results will be saved to: {detect_out_dir}")

        run_detection_pipeline(
            pipeline_config=pipeline_config,
            global_config=global_config,
            input_dir=str(output_folder),
            output_dir=str(detect_out_dir),
        )
    else:
        logger.info("No detection model configured; skipping detection step.")


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

    step_out = run_dir
    step_out.mkdir(parents=True, exist_ok=True)

    pipeline_config.paths.output_dir = str(step_out)

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
            logger.warning("Could not auto-resolve input_dir for filter.")

    run_keypoint_filtering_from_config(
        pipeline_config,
        global_config,
        output_dir=step_out,
    )


def _handle_assess_command(
    args, pipeline_config: PipelineConfig, global_config: GlobalConfig
):
    from evaluation_pipeline import run_pose_assessment_pipeline

    input_dir, base_pipeline_out = get_pipeline_io_paths(
        global_config.paths, pipeline_config.paths.dataset
    )

    # Check if enhanced detection results exist from previous pipeline step
    enhanced_detection_dir = _find_enhanced_detection_results_dir(
        args.pipeline_name, global_config
    )

    if enhanced_detection_dir:
        logger.info(
            f"Found enhanced detection results, using them for evaluation: {enhanced_detection_dir}"
        )
        run_dir = make_run_dir(
            base_out=base_pipeline_out,
            pipeline_name=f"{args.pipeline_name}_enhanced",
            step_name=args.command,
            cfg_path=args.pipeline_config,
            global_config_obj=global_config,
            pipeline_config_obj=pipeline_config,
        )
    else:
        logger.info(
            "No enhanced detection results found, using regular evaluation")
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

    if pipeline_config.evaluation.input_dir:
        logger.info(
            f"Using manually specified input_dir: {pipeline_config.evaluation.input_dir}"
        )
        run_pose_assessment_pipeline(
            pipeline_config,
            global_config,
            output_dir=step_out,
            input_dir=pipeline_config.evaluation.input_dir,
        )
        return

    step_candidates = ["detect", "noise", "filter"]

    # If enhanced detection results exist, prioritize them
    if enhanced_detection_dir:
        # Evaluate the enhanced detection results
        step_eval_dir = step_out / "enhanced_detect"
        step_eval_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Running evaluation for enhanced detection, using input_dir: {enhanced_detection_dir}"
        )
        pipeline_config.evaluation.input_dir = enhanced_detection_dir
        run_pose_assessment_pipeline(
            pipeline_config,
            global_config,
            output_dir=step_eval_dir,
            input_dir=enhanced_detection_dir,
        )
    else:
        # Fall back to regular step evaluation
        for step in step_candidates:
            step_dir = os.path.join(
                base_pipeline_out, args.pipeline_name, step)
            if os.path.exists(step_dir):
                step_eval_dir = step_out / step
                step_eval_dir.mkdir(parents=True, exist_ok=True)

                logger.info(
                    f"Running evaluation for step: {step}, using input_dir: {step_dir}"
                )
                pipeline_config.evaluation.input_dir = step_dir
                run_pose_assessment_pipeline(
                    pipeline_config,
                    global_config,
                    output_dir=step_eval_dir,
                    input_dir=step_dir,
                )
            else:
                logger.warning(
                    f"Step output folder not found: {step_dir}, skipping evaluation."
                )


def _handle_enhance_command(
    args, pipeline_config: PipelineConfig, global_config: GlobalConfig
):
    """Handle the enhance command using the new enhancement pipeline."""
    from enhancement_pipeline import run_enhancement_pipeline

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

    # Use command line overrides if provided
    input_folder = args.input_folder if args.input_folder else input_dir
    output_folder = args.output_folder if args.output_folder else step_out

    if args.input_folder:
        logger.info(f"Overriding input folder: {input_folder}")
    if args.output_folder:
        logger.info(f"Overriding output folder: {output_folder}")

    # Determine enhancement type from config or args (no default)
    enhancement_type = None
    if hasattr(pipeline_config, "enhancement") and hasattr(
        pipeline_config.enhancement, "type"
    ):
        enhancement_type = pipeline_config.enhancement.type
    if hasattr(args, "type") and args.type:
        enhancement_type = args.type
        logger.info(
            f"Using enhancement type from command line: {enhancement_type}")

    if not enhancement_type:
        raise ValueError(
            "Enhancement type must be specified in config or command line")

    # Determine if we should create comparison images (default: True, unless --no_comparison_images is set)
    create_comparison_images = True
    if hasattr(args, "no_comparison_images") and args.no_comparison_images:
        create_comparison_images = False
    elif hasattr(args, "create_comparison_images"):
        create_comparison_images = args.create_comparison_images

    # Determine if we should process as structured dataset (no default)
    dataset_structure = getattr(args, "dataset_structure", False)

    logger.info(f"Enhancement type: {enhancement_type}")
    logger.info(f"Input directory: {input_folder}")
    logger.info(f"Output directory: {output_folder}")
    logger.info(f"Dataset structure mode: {dataset_structure}")
    logger.info(f"Create comparison images: {create_comparison_images}")

    success = run_enhancement_pipeline(
        pipeline_config=pipeline_config,
        global_config=global_config,
        input_dir=input_folder,
        output_dir=output_folder,
        enhancement_type=enhancement_type,
        dataset_structure=dataset_structure,
        create_comparison_images=create_comparison_images,
    )

    if success:
        logger.info("Enhancement pipeline completed successfully")
    else:
        logger.error("Enhancement pipeline completed with errors")


def _get_detection_pipeline_fn(detector_name: str):
    detector_name = detector_name.lower()
    if detector_name == "dwpose":
        from detection_pipeline_DWPose import run_detection_pipeline

        logger.info("Using DWPose detection pipeline.")
    else:
        from detection_pipeline_rtmw import run_detection_pipeline

        logger.info("Using RTMW/MMpose detection pipeline.")
    return run_detection_pipeline

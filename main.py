import argparse
import sys
from utils.config import Config


def run_detect_and_visualize(config):
    from detect_and_visualize_pose import detect_and_visualize_pose

    detect_and_visualize_pose(config)


def run_real_world_noise(input_folder, output_folder):
    import real_world_noise

    real_world_noise.input_folder = input_folder
    real_world_noise.output_folder = output_folder
    real_world_noise.process_all_videos(input_folder, output_folder)


def run_pose_assessment():
    import pose_assessment


def run_filtering(filter_name, filter_kwargs):
    import filtering

    filtering.run_filter(filter_name, filter_kwargs)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Lower Body Pose Estimation Pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Detect & Visualize
    parser_detect = subparsers.add_parser(
        "detect", help="Run detection and visualization pipeline"
    )
    parser_detect.add_argument("--video_folder", type=str, help="Override video folder")
    parser_detect.add_argument(
        "--output_dir", type=str, help="Override output directory"
    )

    # Real-world noise simulation
    parser_noise = subparsers.add_parser(
        "noise", help="Simulate real-world noise on videos"
    )
    parser_noise.add_argument("--input_folder", type=str, help="Input video folder")
    parser_noise.add_argument(
        "--output_folder", type=str, help="Output folder for noisy videos"
    )

    # Pose assessment
    parser_assess = subparsers.add_parser("assess", help="Run pose assessment pipeline")

    # Filtering
    parser_filter = subparsers.add_parser(
        "filter", help="Run keypoint filtering pipeline"
    )
    parser_filter.add_argument(
        "filter_name",
        type=str,
        help="Filter name (gaussian, butterworth, median, etc.)",
    )
    parser_filter.add_argument(
        "--params",
        nargs="*",
        help="Filter parameters as key=value pairs (e.g. sigma=1 window_size=5)",
    )

    # Use argv if provided (for debugging), else sys.argv
    args = parser.parse_args(argv)

    # Load default config
    config = Config()

    if args.command == "detect":
        if args.video_folder:
            config.paths.VIDEO_FOLDER = args.video_folder
        if args.output_dir:
            config.paths.OUTPUT_DIR = args.output_dir
        run_detect_and_visualize(config)

    elif args.command == "noise":
        input_folder = args.input_folder or config.paths.VIDEO_FOLDER
        output_folder = args.output_folder or config.paths.OUTPUT_DIR
        run_real_world_noise(input_folder, output_folder)

    elif args.command == "assess":
        run_pose_assessment()

    elif args.command == "filter":
        filter_kwargs = {}
        if args.params:
            for param in args.params:
                key, value = param.split("=")
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        pass
                filter_kwargs[key] = value
        run_filtering(args.filter_name, filter_kwargs)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    # Set DEBUG_ARGS to a list like you would use in the terminal, e.g.:
    # DEBUG_ARGS = ["detect", "--video_folder", "your/path", "--output_dir", "your/output"]
    DEBUG_ARGS = ["detect"]
    main(DEBUG_ARGS)

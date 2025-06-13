import argparse
from typing import List, Optional
from main_handlers import (
    _handle_detect_command,
    _handle_noise_command,
    _handle_assess_command,
    _handle_filter_command,
)


def parse_main_args(argv: Optional[List[str]] = None):
    # Shared parser for common arguments (like config)
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument(
        "--config", type=str, required=True, help="Path to config YAML file"
    )

    parser = argparse.ArgumentParser(
        description="Lower Body Pose Estimation Pipeline",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    subparsers = parser.add_subparsers(
        dest="command", required=True, help="Available pipeline commands"
    )

    # Subcommand: detect
    parser_detect = subparsers.add_parser(
        "detect", parents=[common_parser], help="Run detection pipeline"
    )
    parser_detect.add_argument("--video_folder", type=str)
    parser_detect.add_argument("--output_dir", type=str)
    parser_detect.set_defaults(func=_handle_detect_command)

    # Subcommand: noise
    parser_noise = subparsers.add_parser(
        "noise", parents=[common_parser], help="Simulate video noise"
    )
    parser_noise.add_argument("--input_folder", type=str)
    parser_noise.add_argument("--output_folder", type=str)
    parser_noise.set_defaults(func=_handle_noise_command)

    # Subcommand: assess
    parser_assess = subparsers.add_parser(
        "assess", parents=[common_parser], help="Run pose assessment"
    )
    parser_assess.set_defaults(func=_handle_assess_command)

    # Subcommand: filter
    parser_filter = subparsers.add_parser(
        "filter", parents=[common_parser], help="Apply keypoint filtering"
    )
    parser_filter.set_defaults(func=_handle_filter_command)

    return parser.parse_args(argv)

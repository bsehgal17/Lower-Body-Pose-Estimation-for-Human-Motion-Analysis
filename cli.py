# cli.py
import argparse
from typing import List, Optional
from main_handlers import (
    _handle_detect_command,
    _handle_noise_command,
    _handle_assess_command,
    _handle_filter_command,
)


def parse_main_args(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(
        description="Lower Body Pose Estimation Pipeline",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--config_file", type=str, default="config.yaml", help="Path to config YAML"
    )

    subparsers = parser.add_subparsers(dest="command", required=True, help="Commands")

    # Subcommand: detect
    parser_detect = subparsers.add_parser("detect", help="Run detection pipeline")
    parser_detect.add_argument("--video_folder", type=str)
    parser_detect.add_argument("--output_dir", type=str)
    parser_detect.set_defaults(func=_handle_detect_command)

    # Subcommand: noise
    parser_noise = subparsers.add_parser("noise", help="Simulate video noise")
    parser_noise.add_argument("--input_folder", type=str)
    parser_noise.add_argument("--output_folder", type=str)
    parser_noise.set_defaults(func=_handle_noise_command)

    # Subcommand: assess
    parser_assess = subparsers.add_parser("assess", help="Run pose assessment")
    parser_assess.set_defaults(func=_handle_assess_command)

    # Subcommand: filter
    parser_filter = subparsers.add_parser("filter", help="Filter keypoints")
    parser_filter.add_argument("filter_name", type=str)
    parser_filter.add_argument(
        "--params", nargs="*", default=[], help="Filter parameters as key=value"
    )
    parser_filter.set_defaults(func=_handle_filter_command)

    return parser.parse_args(argv)

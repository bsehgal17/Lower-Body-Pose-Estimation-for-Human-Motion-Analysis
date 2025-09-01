import argparse
from typing import List, Optional
from main_handlers import (
    _handle_detect_command,
    _handle_noise_command,
    _handle_assess_command,
    _handle_filter_command,
    _handle_enhance_command,
)


def parse_main_args(argv: Optional[List[str]] = None):
    # Shared parser for required config arguments
    common_parser = argparse.ArgumentParser(add_help=False)

    common_parser.add_argument(
        "--pipeline_config",
        type=str,
        required=True,
        help="Path to the pipeline-specific YAML config file",
    )
    common_parser.add_argument(
        "--global_config",
        type=str,
        required=True,
        help="Path to the global YAML config file",
    )
    common_parser.add_argument(
        "--pipeline_name",
        type=str,
        required=True,
        help="Logical name of the full pipeline (used for organizing outputs)",
    )

    # Top-level parser
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
    parser_detect.add_argument(
        "--video_folder", type=str, help="Override input video folder"
    )
    parser_detect.add_argument(
        "--output_dir", type=str, help="Optional custom output dir"
    )
    parser_detect.set_defaults(func=_handle_detect_command)

    # Subcommand: noise
    parser_noise = subparsers.add_parser(
        "noise", parents=[common_parser], help="Simulate noise on input videos"
    )
    parser_noise.add_argument(
        "--input_folder", type=str, help="Optional input override"
    )
    parser_noise.add_argument(
        "--output_folder", type=str, help="Optional output override"
    )
    parser_noise.set_defaults(func=_handle_noise_command)

    # Subcommand: filter
    parser_filter = subparsers.add_parser(
        "filter", parents=[common_parser], help="Apply keypoint filtering to pose data"
    )
    parser_filter.set_defaults(func=_handle_filter_command)

    # Subcommand: assess
    parser_assess = subparsers.add_parser(
        "evaluation", parents=[common_parser], help="Run pose accuracy evaluation"
    )
    parser_assess.set_defaults(func=_handle_assess_command)

    # Subcommand: enhance
    parser_enhance = subparsers.add_parser(
        "enhance",
        parents=[common_parser],
        help="Apply video enhancement (CLAHE, histogram equalization, etc.)",
    )
    parser_enhance.add_argument(
        "--input_folder",
        type=str,
        help="Optional: Override the input folder specified in the config",
    )
    parser_enhance.add_argument(
        "--output_folder",
        type=str,
        help="Optional: Override the output folder specified in the config",
    )
    parser_enhance.add_argument(
        "--enhancement_type",
        type=str,
        choices=["clahe", "histogram_eq", "gaussian_blur", "brightness_adjustment"],
        required=True,
        help="Type of enhancement to apply (required)",
    )
    parser_enhance.add_argument(
        "--dataset_structure",
        action="store_true",
        help="Process videos organized in dataset structure (subject/action/videos)",
    )
    parser_enhance.set_defaults(func=_handle_enhance_command)

    return parser.parse_args(argv)

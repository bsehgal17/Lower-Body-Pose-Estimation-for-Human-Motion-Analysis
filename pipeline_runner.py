# run.py
import os
import sys
import logging
from typing import List, Optional
from cli import parse_main_args
from config.base import get_config

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_pipeline_from_args(argv: Optional[List[str]] = None):
    args = parse_main_args(argv)

    try:
        config_path = os.path.join(os.path.dirname(__file__), args.config_file)
        config = get_config(config_path)
    except Exception as e:
        logger.critical(f"Error loading config: {e}")
        sys.exit(1)

    try:
        args.func(args, config)
    except Exception as e:
        logger.critical(f"Error during '{args.command}': {e}")
        logger.exception("Traceback:")
        sys.exit(1)

    logger.info("Pipeline completed successfully.")


if __name__ == "__main__":
    run_pipeline_from_args()

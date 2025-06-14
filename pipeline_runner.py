import logging
import sys
from typing import List, Optional

from cli import parse_main_args
from config.pipeline_config import get_pipeline_config
from config.global_config import get_global_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_pipeline_from_args(argv: Optional[List[str]] = None):
    # Step 1: Parse CLI arguments
    args = parse_main_args(argv)

    # Step 2: Load pipeline and global config
    try:
        pipeline_config = get_pipeline_config(args.pipeline_config)
        global_config = get_global_config(args.global_config)
    except Exception as e:
        logger.critical(f"Error loading configuration: {e}")
        sys.exit(1)

    # Step 3: Dispatch to command handler
    try:
        args.func(args, pipeline_config, global_config)
    except Exception as e:
        logger.critical(f"Error during '{args.command}': {e}")
        logger.exception("Traceback:")
        sys.exit(1)

    logger.info("Pipeline completed successfully.")


if __name__ == "__main__":
    run_pipeline_from_args()

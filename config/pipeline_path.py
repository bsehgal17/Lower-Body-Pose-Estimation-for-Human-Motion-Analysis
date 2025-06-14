from dataclasses import dataclass
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class PipelinePathsConfig:
    """Pipeline-level paths: dataset name and ground truth file."""

    dataset: str
    ground_truth_file: str

    def __post_init__(self):
        if not os.path.exists(self.ground_truth_file):
            logger.warning(
                f"Ground truth file does not exist: {self.ground_truth_file}")
        logger.info(f"Using dataset: {self.dataset}")

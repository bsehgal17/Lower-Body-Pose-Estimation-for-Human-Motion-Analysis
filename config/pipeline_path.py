from pydantic import BaseModel, model_validator
import logging
import os
from typing import Optional


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PipelinePathsConfig(BaseModel):
    """Pipeline-level paths: dataset name and ground truth file."""
    input_dir: Optional[str] = None
    output_dir: Optional[str] = None
    dataset: str
    ground_truth_file: str

    @model_validator(mode="after")
    def validate_paths(self):
        if not os.path.exists(self.ground_truth_file):
            logger.warning(
                f"Ground truth file does not exist: {self.ground_truth_file}"
            )
        logger.info(f"Using dataset: {self.dataset}")
        return self

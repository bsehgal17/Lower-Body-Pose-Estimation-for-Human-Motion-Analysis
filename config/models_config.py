from pydantic import BaseModel, model_validator
import logging
import os
from typing import Optional


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ModelsConfig(BaseModel):
    """Configuration for pose estimation models (e.g., MMPose)."""

    detector: str
    det_config: Optional[str]
    det_checkpoint: Optional[str]
    pose_config: Optional[str]
    pose_checkpoint: Optional[str]

    @model_validator(mode="after")
    def validate_files(self):
        # Basic validation for local files (checkpoints can be URLs)
        if self.det_config and not os.path.exists(self.det_config):
            logger.warning(f"Detector config file not found: {self.det_config}")
        if self.pose_config and not os.path.exists(self.pose_config):
            logger.warning(f"Pose config file not found: {self.pose_config}")
        return self

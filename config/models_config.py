from dataclasses import dataclass
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class ModelsConfig:
    """Configuration for pose estimation models (e.g., MMPose)."""

    det_config: str
    det_checkpoint: str
    pose_config: str
    pose_checkpoint: str

    def __post_init__(self):
        # Basic validation for local files (checkpoints can be URLs)
        if self.det_config and not os.path.exists(self.det_config):
            logger.warning(
                f"Detector config file not found: {self.det_config}")
        if self.pose_config and not os.path.exists(self.pose_config):
            logger.warning(f"Pose config file not found: {self.pose_config}")

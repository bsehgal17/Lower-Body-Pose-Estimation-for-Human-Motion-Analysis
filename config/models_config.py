from pydantic import BaseModel, model_validator, Field
import logging
import os
from typing import Optional


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ModelsConfig(BaseModel):
    """
    Configuration for pose estimation models and their associated files.

    Defines model architecture, config files, and checkpoint paths for both
    object detection and pose estimation components of the pipeline.
    """

    detector: str = Field(
        ...,
        description="Detector model identifier or architecture name. "
        "Examples: 'faster_rcnn', 'yolox', 'rtmdet', 'ssd'. "
        "Must match available detector configurations in the framework.",
    )

    det_config: Optional[str] = Field(
        default=None,
        description="Path to detector configuration file (.py or .yaml). "
        "Contains model architecture, training parameters, and data pipeline settings. "
        "Can be local file path or URL to config in model zoo.",
    )

    det_checkpoint: Optional[str] = Field(
        default=None,
        description="Path to pre-trained detector model weights (.pth file). "
        "Can be local file path, URL to download, or model zoo identifier. "
        "Must be compatible with the specified detector config.",
    )

    pose_config: Optional[str] = Field(
        default=None,
        description="Path to pose estimation configuration file (.py or .yaml). "
        "Defines keypoint model architecture, joint definitions, and processing pipeline. "
        "Examples: COCO-17, COCO-133, Human3.6M configurations.",
    )

    pose_checkpoint: Optional[str] = Field(
        default=None,
        description="Path to pre-trained pose estimation model weights (.pth file). "
        "Can be local file path, URL, or model zoo identifier. "
        "Must match the pose config architecture and keypoint format.",
    )

    @model_validator(mode="after")
    def validate_files(self):
        # Basic validation for local files (checkpoints can be URLs)
        if self.det_config and not os.path.exists(self.det_config):
            logger.warning(f"Detector config file not found: {self.det_config}")
        if self.pose_config and not os.path.exists(self.pose_config):
            logger.warning(f"Pose config file not found: {self.pose_config}")
        return self

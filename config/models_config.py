from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field
import logging
import os

# Configure logging for the config module
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class ModelsConfig:
    """Configuration for pose estimation models (e.g., MMPose)."""

    det_config: str = "/storage/Projects/Gaitly/bsehgal/Lower-Body-Pose-Estimation-for-Human-Motion-Analysis/mmpose/demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py"
    det_checkpoint: str = "https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"
    pose_config: str = "/storage/Projects/Gaitly/bsehgal/Lower-Body-Pose-Estimation-for-Human-Motion-Analysis/mmpose/projects/rtmpose/rtmpose/wholebody_2d_keypoint/rtmw-x_8xb704-270e_cocktail14-256x192.py"
    pose_checkpoint: str = "/storage/Projects/Gaitly/bsehgal/Lower-Body-Pose-Estimation-for-Human-Motion-Analysis/rtmw-x_simcc-cocktail14_pt-ucoco_270e-256x192-13a2546d_20231208.pth"

    def __post_init__(self):
        # Basic validation for local files (checkpoints can be URLs)
        if self.det_config and not os.path.exists(self.det_config):
            logger.warning(f"Detector config file not found: {self.det_config}")
        if self.pose_config and not os.path.exists(self.pose_config):
            logger.warning(f"Pose config file not found: {self.pose_config}")

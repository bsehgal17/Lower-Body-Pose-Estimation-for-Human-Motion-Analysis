from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class ProcessingConfig:
    """Configuration for general processing parameters."""

    device: str
    nms_threshold: float
    detection_threshold: float
    kpt_threshold: float

    def __post_init__(self):
        if not (0 <= self.nms_threshold <= 1):
            raise ValueError("NMS threshold must be between 0 and 1.")
        if not (0 <= self.detection_threshold <= 1):
            raise ValueError("Detection threshold must be between 0 and 1.")
        if not (0 <= self.kpt_threshold <= 1):
            raise ValueError("Keypoint threshold must be between 0 and 1.")

        if not (self.device.startswith("cuda") or self.device == "cpu"):
            logger.warning(
                f"Invalid device specified: {self.device}. Using 'cpu'.")
            self.device = "cpu"

        if self.device.startswith("cuda"):
            try:
                import torch
                if not torch.cuda.is_available():
                    logger.warning(
                        "CUDA requested but not available. Falling back to CPU.")
                    self.device = "cpu"
            except ImportError:
                logger.warning("PyTorch not installed, falling back to CPU.")
                self.device = "cpu"

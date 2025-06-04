from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field
import logging

# Configure logging for the config module
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class ProcessingConfig:
    """Configuration for general processing parameters."""

    device: str = "cuda:0"
    nms_threshold: float = 0.3
    detection_threshold: float = 0.3
    kpt_threshold: float = 0.3
    # Add other processing parameters here, e.g.,
    # filter_type: str = "moving_average"
    # filter_window_size: int = 5
    # noise_type: Optional[str] = None
    # noise_intensity: Optional[float] = 0.1

    def __post_init__(self):
        if not (0 <= self.nms_threshold <= 1):
            raise ValueError("NMS threshold must be between 0 and 1.")
        if not (0 <= self.detection_threshold <= 1):
            raise ValueError("Detection threshold must be between 0 and 1.")
        if not (0 <= self.kpt_threshold <= 1):
            raise ValueError("Keypoint threshold must be between 0 and 1.")
        if not (self.device.startswith("cuda") or self.device == "cpu"):
            logger.warning(f"Invalid device specified: {self.device}. Using 'cpu'.")
            self.device = "cpu"
        # Check if cuda is available if 'cuda' device is requested
        if self.device.startswith("cuda"):
            try:
                import torch  # Assuming PyTorch for CUDA check

                if not torch.cuda.is_available():
                    logger.warning(
                        "CUDA device requested but not available. Falling back to CPU."
                    )
                    self.device = "cpu"
            except ImportError:
                logger.warning(
                    "PyTorch not installed, cannot check CUDA availability. Falling back to CPU."
                )
                self.device = "cpu"

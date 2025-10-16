from pydantic import BaseModel, model_validator, Field
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ProcessingConfig(BaseModel):
    """
    Configuration for processing parameters used during pose estimation inference.

    Controls various thresholds and hardware settings that affect the quality
    and performance of detection and pose estimation pipelines.
    """

    device: str = Field(
        ...,
        description="Computing device for model inference. Options: 'cpu', 'cuda', 'cuda:0', etc. "
        "CUDA devices require PyTorch with CUDA support and compatible GPU hardware. "
        "Automatically falls back to CPU if CUDA is unavailable.",
    )

    nms_threshold: float = Field(
        ...,
        ge=0,
        le=1,
        description="Non-Maximum Suppression threshold for object detection [0.0-1.0]. "
        "Lower values = more aggressive suppression of overlapping detections. "
        "Typical range: 0.3-0.7. Higher values may result in duplicate detections, "
        "lower values may suppress valid detections.",
    )

    detection_threshold: float = Field(
        ...,
        ge=0,
        le=1,
        description="Minimum confidence threshold for object detection [0.0-1.0]. "
        "Only detections with confidence >= this value are kept. "
        "Typical range: 0.3-0.8. Higher values = fewer but more confident detections, "
        "lower values = more detections but potentially more false positives.",
    )

    kpt_threshold: float = Field(
        ...,
        ge=0,
        le=1,
        description="Minimum confidence threshold for individual keypoints [0.0-1.0]. "
        "Keypoints with visibility score < this value may be filtered or marked invalid. "
        "Typical range: 0.1-0.5. Affects pose quality vs completeness trade-off.",
    )

    @model_validator(mode="after")
    def validate_device(self):
        if not (self.device.startswith("cuda") or self.device == "cpu"):
            logger.warning(f"Invalid device specified: {self.device}. Using 'cpu'.")
            self.device = "cpu"

        if self.device.startswith("cuda"):
            try:
                import torch

                if not torch.cuda.is_available():
                    logger.warning(
                        "CUDA requested but not available. Falling back to CPU."
                    )
                    self.device = "cpu"
            except ImportError:
                logger.warning("PyTorch not installed, falling back to CPU.")
                self.device = "cpu"
        return self

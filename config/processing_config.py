from pydantic import BaseModel, model_validator, Field
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ProcessingConfig(BaseModel):
    """Configuration for general processing parameters."""

    device: str
    nms_threshold: float = Field(ge=0, le=1)
    detection_threshold: float = Field(ge=0, le=1)
    kpt_threshold: float = Field(ge=0, le=1)

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

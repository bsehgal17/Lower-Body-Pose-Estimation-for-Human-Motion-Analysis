from dataclasses import dataclass
from typing import Optional


@dataclass
class ConfidenceFilteringConfig:
    """Configuration for confidence-based skeleton filtering."""

    # Minimum confidence thresholds for skeleton filtering
    min_bbox_confidence: float
    min_keypoint_confidence: float

    # Weights for overall confidence score calculation
    bbox_weight: float
    keypoint_weight: float

    # Minimum number of valid keypoints required for a skeleton
    min_valid_keypoints: int

    # Enable/disable confidence filtering
    enabled: bool

    def __post_init__(self):
        """Validate configuration values after initialization."""
        # Normalize weights to ensure they sum to 1.0
        total_weight = self.bbox_weight + self.keypoint_weight
        if total_weight != 1.0:
            self.bbox_weight = self.bbox_weight / total_weight
            self.keypoint_weight = self.keypoint_weight / total_weight

        # Validate ranges
        if not (0.0 <= self.min_bbox_confidence <= 1.0):
            raise ValueError("min_bbox_confidence must be between 0.0 and 1.0")

        if not (0.0 <= self.min_keypoint_confidence <= 1.0):
            raise ValueError("min_keypoint_confidence must be between 0.0 and 1.0")

        if self.min_valid_keypoints < 0:
            raise ValueError("min_valid_keypoints must be non-negative")

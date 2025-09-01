"""
Metric extractors for video analysis.
"""

from base_classes import BaseMetricExtractor
import cv2
import numpy as np
from typing import List


class BrightnessExtractor(BaseMetricExtractor):
    """Extracts brightness data from video frames."""

    def extract(self) -> List[float]:
        """Extract average brightness (L-channel) for each frame."""
        if not self.validate_video():
            return []

        cap = cv2.VideoCapture(self.video_path)
        brightness_per_frame = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            avg_l_channel = np.mean(lab[:, :, 0])
            brightness_per_frame.append(avg_l_channel)

        cap.release()
        return brightness_per_frame


class ContrastExtractor(BaseMetricExtractor):
    """Extracts contrast data from video frames."""

    def extract(self) -> List[float]:
        """Extract contrast (standard deviation of L-channel) for each frame."""
        if not self.validate_video():
            return []

        cap = cv2.VideoCapture(self.video_path)
        contrast_per_frame = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            std_l_channel = np.std(lab[:, :, 0])
            contrast_per_frame.append(std_l_channel)

        cap.release()
        return contrast_per_frame


class SharpnessExtractor(BaseMetricExtractor):
    """Extracts sharpness data from video frames using Laplacian variance."""

    def extract(self) -> List[float]:
        """Extract sharpness (Laplacian variance) for each frame."""
        if not self.validate_video():
            return []

        cap = cv2.VideoCapture(self.video_path)
        sharpness_per_frame = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_per_frame.append(laplacian_var)

        cap.release()
        return sharpness_per_frame


class MetricExtractorFactory:
    """Factory for creating metric extractors."""

    _extractors = {
        "brightness": BrightnessExtractor,
        "contrast": ContrastExtractor,
        "sharpness": SharpnessExtractor,
    }

    @classmethod
    def create_extractor(cls, metric_name: str, video_path: str) -> BaseMetricExtractor:
        """Create a metric extractor for the specified metric."""
        metric_name = metric_name.lower()

        if metric_name not in cls._extractors:
            raise ValueError(
                f"Unknown metric: {metric_name}. Available metrics: {list(cls._extractors.keys())}"
            )

        return cls._extractors[metric_name](video_path)

    @classmethod
    def register_extractor(cls, metric_name: str, extractor_class: type):
        """Register a new metric extractor."""
        if not issubclass(extractor_class, BaseMetricExtractor):
            raise ValueError("Extractor class must inherit from BaseMetricExtractor")

        cls._extractors[metric_name.lower()] = extractor_class

    @classmethod
    def get_available_metrics(cls) -> List[str]:
        """Get list of available metrics."""
        return list(cls._extractors.keys())

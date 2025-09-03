"""
Contrast extraction from video frames.
"""

import cv2
import numpy as np
from typing import List
from ..base_classes import BaseMetricExtractor


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

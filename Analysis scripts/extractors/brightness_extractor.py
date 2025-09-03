"""
Brightness extraction from video frames.
"""

import cv2
import numpy as np
from typing import List
from base_classes import BaseMetricExtractor


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

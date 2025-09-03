"""
Sharpness extraction from video frames.
"""

import cv2
from typing import List
from base_classes import BaseMetricExtractor


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

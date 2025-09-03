"""
Video file discovery and path management.
"""

import os
import pandas as pd
from base_classes import BaseDataProcessor


class VideoPathResolver(BaseDataProcessor):
    """Resolves video file paths for different datasets."""

    def find_video_for_row(self, video_row: pd.Series) -> str:
        """Find video file path for a given data row."""
        if self.config.name.lower() == "humaneva":
            return self._find_humaneva_video(video_row)
        elif self.config.name.lower() == "movi":
            return self._find_movi_video(video_row)
        else:
            raise ValueError(f"Unknown dataset: {self.config.name}")

    def _find_humaneva_video(self, video_row: pd.Series) -> str:
        """Find video file for HumanEva dataset."""
        subject = video_row.get(self.config.subject_column)
        action = video_row.get(self.config.action_column)
        camera = video_row.get(self.config.camera_column)

        if not all([subject, action, camera]):
            return None

        video_filename = f"{subject}_{action}_C{camera}.avi"
        video_path = os.path.join(
            self.config.video_directory, subject, action, video_filename
        )

        return video_path if os.path.exists(video_path) else None

    def _find_movi_video(self, video_row: pd.Series) -> str:
        """Find video file for MoVi dataset."""
        subject = video_row.get(self.config.subject_column)

        if not subject:
            return None

        for root, dirs, files in os.walk(self.config.video_directory):
            for file in files:
                if file.endswith(".mp4") and subject in file:
                    return os.path.join(root, file)

        return None

    def process(self, *args, **kwargs) -> pd.DataFrame:
        """Process method required by base class."""
        # This processor doesn't return DataFrames directly
        raise NotImplementedError("Use find_video_for_row method instead")

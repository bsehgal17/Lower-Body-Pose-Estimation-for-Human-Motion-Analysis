"""
Frame synchronization utilities.
"""

from typing import Dict, Any
from ..base_classes import BaseDataProcessor


class FrameSynchronizer(BaseDataProcessor):
    """Handles frame synchronization for multi-camera datasets."""

    def get_synced_start_frame(self, video_row_data: Dict[str, Any]) -> int:
        """Get the synchronized start frame for a video."""
        synced_start_frame = 0

        if hasattr(self.config, "sync_data") and self.config.sync_data:
            try:
                subject_key = str(video_row_data.get(self.config.subject_column))
                action_key = video_row_data.get(self.config.action_column)

                if action_key and isinstance(action_key, str):
                    action_key = action_key.replace("_", " ").title()

                if self.config.camera_column:
                    camera_id = video_row_data.get(self.config.camera_column)
                    camera_index = int(camera_id) - 1
                    synced_start_frame = self.config.sync_data.data[subject_key][
                        action_key
                    ][camera_index]

                if synced_start_frame < 0:
                    synced_start_frame = 0

            except (KeyError, IndexError, TypeError):
                synced_start_frame = 0

        return synced_start_frame

    def process(self, *args, **kwargs):
        """Process method required by base class."""
        # This processor doesn't return DataFrames directly
        raise NotImplementedError("Use get_synced_start_frame method instead")

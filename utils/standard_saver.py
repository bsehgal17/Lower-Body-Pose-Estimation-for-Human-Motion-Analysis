import os
import json
import pickle
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import cv2
import numpy as np
from utils.video_format_utils import get_video_format_info

logger = logging.getLogger(__name__)


@dataclass
class SaveConfig:
    """Configuration for saving data."""

    output_dir: str
    relative_subdir: Optional[str] = None
    save_json: bool = True
    save_pickle: bool = True
    save_video_overlay: bool = False
    video_input_dir: Optional[str] = None

    def get_full_output_dir(self) -> str:
        """Get the complete output directory path."""
        if self.relative_subdir:
            return os.path.join(self.output_dir, self.relative_subdir)
        return self.output_dir


class StandardDataSaver:
    """
    A standardized data saver that can handle keypoint data, detection configs,
    and optional video overlay generation for multiple processing pipelines.
    """

    def __init__(self, save_config: SaveConfig):
        self.config = save_config

    def save_data(
        self,
        data: Dict[str, Any],
        original_file_path: str,
        suffix: str = "",
        video_name: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Save data in JSON and/or pickle format, optionally with video overlay.

        Args:
            data: Data dictionary to save (should contain standardized format)
            original_file_path: Path to the original file (for naming reference)
            suffix: Optional suffix to add to output filename
            video_name: Optional video name (if None, will try to extract from data)

        Returns:
            Dictionary with paths of saved files
        """
        # Extract video name from data if not provided
        if video_name is None:
            video_name = self._extract_video_name(data, original_file_path)

        # Create output directory
        output_dir = self.config.get_full_output_dir()
        os.makedirs(output_dir, exist_ok=True)

        # Generate base filename
        original_basename = os.path.basename(original_file_path)
        base_name = os.path.splitext(original_basename)[0]
        if suffix:
            base_name = f"{base_name}_{suffix}"

        saved_paths = {}

        # Save JSON
        if self.config.save_json:
            json_path = os.path.join(output_dir, f"{base_name}.json")
            self._save_as_json(data, json_path)
            saved_paths["json"] = json_path

        # Save pickle
        if self.config.save_pickle:
            pkl_path = os.path.join(output_dir, f"{base_name}.pkl")
            self._save_as_pickle(data, pkl_path)
            saved_paths["pickle"] = pkl_path

        # Save video overlay if requested
        if self.config.save_video_overlay and video_name:
            video_path = self._find_video_file(video_name, original_file_path)
            if video_path:
                overlay_path = os.path.join(output_dir, f"{base_name}_overlay")
                self._create_video_overlay(video_path, data, overlay_path)
                saved_paths["video"] = overlay_path

        return saved_paths

    def _extract_video_name(
        self, data: Dict[str, Any], original_file_path: str
    ) -> Optional[str]:
        """Extract video name from data or file path."""
        # Try to get from data structure
        if "video_name" in data:
            return data["video_name"]

        # Try to derive from persons data (if available)
        if "persons" in data and data["persons"]:
            # Could implement logic to extract from filename patterns
            pass

        # Fallback: use original file basename
        base_name = os.path.splitext(os.path.basename(original_file_path))[0]
        return base_name

    def _save_as_json(self, data: Dict[str, Any], file_path: str):
        """Save data as JSON file."""
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)
        logger.info(f"Data saved to JSON: {file_path}")

    def _save_as_pickle(self, data: Dict[str, Any], file_path: str):
        """Save data as pickle file."""
        with open(file_path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"Data saved to pickle: {file_path}")

    def _find_video_file(
        self, video_name: str, original_file_path: str
    ) -> Optional[str]:
        """Find the corresponding video file for overlay generation."""
        if self.config.video_input_dir:
            search_dir = self.config.video_input_dir
        else:
            # Search in the same directory as the original file
            search_dir = os.path.dirname(original_file_path)

        # Common video extensions
        video_extensions = [".avi", ".mp4", ".mov", ".mkv", ".flv", ".wmv"]

        for ext in video_extensions:
            video_path = os.path.join(search_dir, f"{video_name}{ext}")
            if os.path.exists(video_path):
                return video_path

        # Also try replacing .json with video extensions in original path
        base_path = os.path.splitext(original_file_path)[0]
        for ext in video_extensions:
            video_path = f"{base_path}{ext}"
            if os.path.exists(video_path):
                return video_path

        logger.warning(f"Video file not found for overlay: {video_name}")
        return None

    def _create_video_overlay(
        self, video_path: str, data: Dict[str, Any], output_path_base: str
    ):
        """
        Create video overlay with keypoints.

        Args:
            video_path: Path to input video
            data: Data containing keypoint information
            output_path_base: Base path for output video (extension will be added)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video {video_path}")
            return

        # Video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Get fourcc and extension from input video
        fourcc, input_extension = get_video_format_info(video_path)

        # Update output path to use same extension
        output_path_with_ext = f"{output_path_base}{input_extension}"
        out = cv2.VideoWriter(output_path_with_ext, fourcc, fps, (width, height))

        # Convert data to frame-based structure for easier processing
        frame_keypoints = self._extract_frame_keypoints(data)

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Draw keypoints for this frame
            if frame_idx in frame_keypoints:
                for person_keypoints in frame_keypoints[frame_idx]:
                    for joint in person_keypoints:
                        if len(joint) >= 2:  # Ensure we have x, y coordinates
                            x, y = int(joint[0]), int(joint[1])
                            if not np.isnan(x) and not np.isnan(y):
                                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            out.write(frame)
            frame_idx += 1

        cap.release()
        out.release()
        logger.info(f"Video overlay saved to: {output_path_with_ext}")

    def _extract_frame_keypoints(
        self, data: Dict[str, Any]
    ) -> Dict[int, List[List[List[float]]]]:
        """
        Extract keypoints organized by frame index.

        Returns:
            Dictionary mapping frame_idx -> list of person keypoints
        """
        frame_keypoints = {}

        # Handle different data formats
        if "persons" in data:
            # Standard format with persons
            for person in data["persons"]:
                if "poses" in person:
                    for pose in person["poses"]:
                        frame_idx = pose["frame_idx"]
                        keypoints = pose["keypoints"]

                        if frame_idx not in frame_keypoints:
                            frame_keypoints[frame_idx] = []
                        frame_keypoints[frame_idx].append(keypoints)

        elif "keypoints" in data:
            # Legacy format - keypoints is a list of frames
            keypoints_frames = data["keypoints"]
            for frame_idx, frame_data in enumerate(keypoints_frames):
                if "keypoints" in frame_data:
                    frame_keypoints[frame_idx] = frame_data["keypoints"]

        return frame_keypoints


# Convenience functions for common use cases
def save_standard_format(
    data: Dict[str, Any],
    output_dir: str,
    original_file_path: str,
    suffix: str = "",
    relative_subdir: Optional[str] = None,
    save_json: bool = True,
    save_pickle: bool = True,
    save_video_overlay: bool = False,
    video_input_dir: Optional[str] = None,
    video_name: Optional[str] = None,
) -> Dict[str, str]:
    """
    Convenience function to save data in standard format.

    Args:
        data: Data to save
        output_dir: Base output directory
        original_file_path: Original file path for naming reference
        suffix: Optional suffix for filename
        relative_subdir: Optional subdirectory within output_dir
        save_json: Whether to save as JSON
        save_pickle: Whether to save as pickle
        save_video_overlay: Whether to create video overlay
        video_input_dir: Directory to search for video files
        video_name: Optional video name (extracted from data if None)

    Returns:
        Dictionary with paths of saved files
    """
    config = SaveConfig(
        output_dir=output_dir,
        relative_subdir=relative_subdir,
        save_json=save_json,
        save_pickle=save_pickle,
        save_video_overlay=save_video_overlay,
        video_input_dir=video_input_dir,
    )

    saver = StandardDataSaver(config)
    return saver.save_data(data, original_file_path, suffix, video_name)


def extract_video_name_from_path_structure(json_path: str) -> str:
    """
    Extract video name from file path structure.
    Useful for standardized directory structures.
    """
    # Convert to Path object for easier manipulation
    path_obj = Path(json_path)

    # Remove .json extension to get base name
    base_name = path_obj.stem

    return base_name


def create_relative_subdir_from_path(
    json_path: str, anchor_prefix: str = "S"
) -> Optional[str]:
    """
    Create relative subdirectory path from file structure.

    Args:
        json_path: Original JSON file path
        anchor_prefix: Prefix to look for as anchor point (e.g., "S" for "S1", "S2", etc.)

    Returns:
        Relative subdirectory path or None if anchor not found
    """
    try:
        json_path_obj = Path(json_path)

        # Find the anchor index (e.g., "S1", "S2", etc.)
        anchor_index = next(
            i
            for i, part in enumerate(json_path_obj.parts)
            if part.startswith(anchor_prefix)
        )

        # Construct relative path from anchor up to parent of .json file
        relative_subdir = Path(*json_path_obj.parts[anchor_index:-1])

        return str(relative_subdir)
    except (StopIteration, IndexError):
        logger.warning(
            f"Could not find anchor with prefix '{anchor_prefix}' in path: {json_path}"
        )
        return None

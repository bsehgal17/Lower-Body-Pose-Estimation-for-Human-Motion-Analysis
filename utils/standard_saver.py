import os
import json
import pickle
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
import cv2
import numpy as np
from utils.video_format_utils import get_video_format_info
from utils.data_structures import VideoData

logger = logging.getLogger(__name__)


@dataclass
class SaveConfig:
    """Configuration for saving data."""

    output_dir: str
    relative_subdir: Optional[str]
    save_json: bool
    save_pickle: bool
    save_video_overlay: bool
    video_input_dir: Optional[str]

    def get_full_output_dir(self) -> str:
        """Get the complete output directory path."""
        if self.relative_subdir:
            return os.path.join(self.output_dir, self.relative_subdir)
        return self.output_dir


@dataclass
class SavedData:
    """
    Structured representation of data being saved to JSON/PKL files.
    This dataclass encapsulates the complete content structure.
    """

    video_data: VideoData
    detection_config: Optional[Dict[str, Any]]
    processing_metadata: Optional[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = self.video_data.to_dict()

        # Add any additional metadata
        if self.detection_config:
            result["detection_config"] = self.detection_config

        if self.processing_metadata:
            result["processing_metadata"] = self.processing_metadata

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SavedData":
        """Create SavedData from dictionary."""
        video_data = VideoData.from_dict(data)

        return cls(
            video_data=video_data,
            detection_config=data.get("detection_config"),
            processing_metadata=data.get("processing_metadata"),
        )

    @classmethod
    def from_video_data(
        cls,
        video_data: VideoData,
        detection_config: Optional[Dict[str, Any]] = None,
        processing_metadata: Optional[Dict[str, Any]] = None,
    ) -> "SavedData":
        """Create SavedData from VideoData and optional metadata."""
        return cls(
            video_data=video_data,
            detection_config=detection_config,
            processing_metadata=processing_metadata,
        )

    @classmethod
    def from_legacy_dict(cls, data: Dict[str, Any], video_name: str) -> "SavedData":
        """
        Create SavedData from legacy dictionary format.
        This helps with backward compatibility.
        """
        # Check if it's already in the new format
        if "video_name" in data and "persons" in data:
            return cls.from_dict(data)

        # Convert legacy format to new format
        video_data = VideoData(video_name=video_name)

        # Handle legacy "keypoints" format
        if "keypoints" in data:
            for frame_idx, frame_data in enumerate(data["keypoints"]):
                if "keypoints" in frame_data:
                    for person_idx, person_data in enumerate(frame_data["keypoints"]):
                        person = video_data.get_or_create_person(person_idx)

                        if "keypoints" in person_data:
                            keypoints = person_data["keypoints"]
                            keypoints_visible = person_data.get("keypoints_visible", [])
                            bbox = person_data.get("bbox", [])
                            bbox_scores = person_data.get("bbox_scores", [])

                            if bbox:
                                person.add_detection(frame_idx, bbox, 1.0, label=0)

                            person.add_pose(
                                frame_idx,
                                keypoints,
                                keypoints_visible,
                                bbox,
                                bbox_scores,
                            )

        return cls(
            video_data=video_data,
            detection_config=data.get("detection_config"),
            processing_metadata=data.get("processing_metadata"),
        )


class StandardDataSaver:
    """
    A standardized data saver that can handle keypoint data, detection configs,
    and optional video overlay generation for multiple processing pipelines.
    """

    def __init__(self, save_config: SaveConfig):
        self.config = save_config

    def save_data(
        self,
        data: Union[SavedData, Dict[str, Any]],
        original_file_path: str,
        suffix: str = "",
        video_name: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Save data in JSON and/or pickle format, optionally with video overlay.

        Args:
            data: SavedData dataclass or legacy dictionary to save
            original_file_path: Path to the original file (for naming reference)
            suffix: Optional suffix to add to output filename
            video_name: Optional video name (if None, will try to extract from data)

        Returns:
            Dictionary with paths of saved files
        """
        # Convert to SavedData if it's a legacy dictionary
        if isinstance(data, dict):
            if video_name is None:
                video_name = self._extract_video_name_from_dict(
                    data, original_file_path
                )
            saved_data = SavedData.from_legacy_dict(data, video_name)
        else:
            saved_data = data
            if video_name is None:
                video_name = saved_data.video_data.video_name

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
            self._save_as_json(saved_data, json_path)
            saved_paths["json"] = json_path

        # Save pickle
        if self.config.save_pickle:
            pkl_path = os.path.join(output_dir, f"{base_name}.pkl")
            self._save_as_pickle(saved_data, pkl_path)
            saved_paths["pickle"] = pkl_path

        # Save video overlay if requested
        if self.config.save_video_overlay and video_name:
            video_path = self._find_video_file(video_name, original_file_path)
            if video_path:
                overlay_path = os.path.join(output_dir, f"{base_name}")
                self._create_video_overlay(video_path, saved_data, overlay_path)
                saved_paths["video"] = overlay_path

        return saved_paths

    def _extract_video_name_from_dict(
        self, data: Dict[str, Any], original_file_path: str
    ) -> str:
        """Extract video name from legacy dictionary data or file path."""
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

    def _extract_video_name(
        self, data: Union[SavedData, Dict[str, Any]], original_file_path: str
    ) -> Optional[str]:
        """Extract video name from SavedData or legacy dictionary."""
        if isinstance(data, SavedData):
            return data.video_data.video_name
        else:
            return self._extract_video_name_from_dict(data, original_file_path)

    def _save_as_json(self, data: SavedData, file_path: str):
        """Save SavedData as JSON file."""
        data_dict = data.to_dict()
        with open(file_path, "w") as f:
            json.dump(data_dict, f, indent=4)
        logger.info(f"Data saved to JSON: {file_path}")

    def _save_as_pickle(self, data: SavedData, file_path: str):
        """Save SavedData as pickle file in dictionary format for compatibility."""
        # Save as dictionary for backward compatibility with evaluation scripts
        data_dict = data.to_dict()
        with open(file_path, "wb") as f:
            pickle.dump(data_dict, f)
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
        self, video_path: str, data: SavedData, output_path_base: str
    ):
        """
        Create video overlay with keypoints.

        Args:
            video_path: Path to input video
            data: SavedData containing keypoint information
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
        frame_keypoints = self._extract_frame_keypoints_from_saved_data(data)

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Draw keypoints for this frame
            if frame_idx in frame_keypoints:
                for person_keypoints in frame_keypoints[frame_idx]:
                    for joint in person_keypoints:
                        for i in range(0, len(joint)):
                            if len(joint[i]) >= 2:  # Ensure we have x, y coordinates
                                x, y = int(joint[i][0]), int(joint[i][1])
                                if not np.isnan(x) and not np.isnan(y):
                                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            out.write(frame)
            frame_idx += 1

        cap.release()
        out.release()
        logger.info(f"Video overlay saved to: {output_path_with_ext}")

    def _extract_frame_keypoints_from_saved_data(
        self, data: SavedData
    ) -> Dict[int, List[List[List[float]]]]:
        """
        Extract keypoints organized by frame index from SavedData.

        Returns:
            Dictionary mapping frame_idx -> list of person keypoints
        """
        frame_keypoints = {}

        for person in data.video_data.persons:
            for pose in person.poses:
                frame_idx = pose.frame_idx
                keypoints = pose.keypoints

                if frame_idx not in frame_keypoints:
                    frame_keypoints[frame_idx] = []
                frame_keypoints[frame_idx].append(keypoints)

        return frame_keypoints

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
    data: Union[SavedData, VideoData, Dict[str, Any]],
    output_dir: str,
    original_file_path: str,
    suffix: str = "",
    relative_subdir: Optional[str] = None,
    save_json: bool = True,
    save_pickle: bool = True,
    save_video_overlay: bool = False,
    video_input_dir: Optional[str] = None,
    video_name: Optional[str] = None,
    detection_config: Optional[Dict[str, Any]] = None,
    processing_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """
    Convenience function to save data in standard format.

    Args:
        data: SavedData, VideoData, or legacy dictionary to save
        output_dir: Base output directory
        original_file_path: Original file path for naming reference
        suffix: Optional suffix for filename
        relative_subdir: Optional subdirectory within output_dir
        save_json: Whether to save as JSON
        save_pickle: Whether to save as pickle
        save_video_overlay: Whether to create video overlay
        video_input_dir: Directory to search for video files
        video_name: Optional video name (extracted from data if None)
        detection_config: Optional detection configuration metadata
        processing_metadata: Optional processing metadata

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

    # Convert data to SavedData if needed
    if isinstance(data, VideoData):
        # Convert VideoData to SavedData
        saved_data = SavedData.from_video_data(
            data, detection_config, processing_metadata
        )
    elif isinstance(data, dict):
        # Keep legacy dictionary support
        saved_data = data
    else:
        # Already SavedData
        saved_data = data

    return saver.save_data(saved_data, original_file_path, suffix, video_name)


def load_saved_data(file_path: str) -> SavedData:
    """
    Load SavedData from either JSON or PKL file.

    Args:
        file_path: Path to the .json or .pkl file

    Returns:
        SavedData instance

    Raises:
        ValueError: If file format is not supported
    """
    if file_path.endswith(".json"):
        with open(file_path, "r") as f:
            data_dict = json.load(f)
        return SavedData.from_dict(data_dict)
    elif file_path.endswith(".pkl"):
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, SavedData):
            return data
        elif isinstance(data, dict):
            # Handle legacy pkl files
            video_name = data.get("video_name", "unknown")
            return SavedData.from_legacy_dict(data, video_name)
        else:
            raise ValueError(f"Unsupported data type in PKL file: {type(data)}")
    else:
        raise ValueError(f"Unsupported file format: {file_path}")


def create_saved_data_from_poses(
    video_name: str,
    persons_data: List[Dict[str, Any]],
    detection_config: Optional[Dict[str, Any]] = None,
    processing_metadata: Optional[Dict[str, Any]] = None,
) -> SavedData:
    """
    Create SavedData from raw poses data.

    Args:
        video_name: Name of the video
        persons_data: List of person data dictionaries
        detection_config: Optional detection configuration
        processing_metadata: Optional processing metadata

    Returns:
        SavedData instance
    """
    video_data = VideoData(video_name=video_name)

    for person_data in persons_data:
        person_id = person_data.get("person_id", 0)
        person = video_data.get_or_create_person(person_id)

        # Add detections
        for detection_data in person_data.get("detections", []):
            person.add_detection(
                detection_data["frame_idx"],
                detection_data["bbox"],
                detection_data["score"],
                detection_data["label"],
            )

        # Add poses
        for pose_data in person_data.get("poses", []):
            person.add_pose(
                pose_data["frame_idx"],
                pose_data["keypoints"],
                pose_data["keypoints_visible"],
                pose_data["bbox"],
                pose_data["bbox_scores"],
            )

    return SavedData.from_video_data(video_data, detection_config, processing_metadata)


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

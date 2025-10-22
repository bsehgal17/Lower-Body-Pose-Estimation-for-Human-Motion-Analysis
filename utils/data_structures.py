from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
import numpy as np


class Detection(BaseModel):
    """
    Single detection data for a frame.

    Contains bounding box and confidence information for an object detected
    in a specific video frame using object detection models.
    """

    detection_frame_idx: int = Field(
        ...,
        ge=0,
        description="Zero-based frame index where this object detection occurred in the video sequence. "
        "Zero-based indexing: first frame = 0, second frame = 1, etc. "
        "For a video with N frames, valid indices are 0 to (N-1). "
        "This frame index corresponds to when the object detector found this bbox.",
    )
    detection_bbox: List[float] = Field(
        ...,
        min_items=4,
        max_items=4,
        description="Object detection bounding box coordinates in pixels [x1, y1, x2, y2] format. "
        "This is the raw output from the object detector (e.g., YOLO, Faster R-CNN) "
        "identifying the general object location. x1,y1 = top-left, x2,y2 = bottom-right. "
        "Zero-based pixel coordinates: top-left image corner is (0,0), x increases right, y down.",
    )
    detection_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Object detection confidence score between 0.0 and 1.0. "
        "Indicates how confident the detector is that this bbox contains the specified object class. "
        "Higher values = more confident detection. Threshold typically 0.3-0.7 for filtering.",
    )
    object_label: int = Field(
        ...,
        ge=0,
        description="Object class label ID. Typically 0 for 'person' in COCO dataset format. "
        "Other common labels: 1=bicycle, 2=car, etc.",
    )


class PoseData(BaseModel):
    """
    Pose estimation data for a single frame.

    Contains keypoint coordinates, visibility scores, and associated bounding box
    information for human pose estimation results in a specific video frame.
    """

    frame_idx: int = Field(
        ...,
        ge=0,
        description="Zero-based frame index where pose estimation was performed in the video sequence. "
        "Zero-based indexing: first frame = 0, second frame = 1, third frame = 2, etc. "
        "This frame index corresponds to when keypoint coordinates were estimated. "
        "May not match detection frame_idx if pose estimation is done on subset of frames.",
    )
    keypoints: List[List[Union[float, List[float]]]] = Field(
        ...,
        description="2D keypoint coordinates in pixels. Shape: (num_joints, 2) where each joint "
        "has [x, y] coordinates in image pixel space. Uses zero-based pixel coordinates: "
        "top-left corner of image is (0.0, 0.0), x increases rightward, y increases downward. "
        "Joint order depends on the pose estimation model used (e.g., COCO-17, COCO-133, etc.). "
        "Coordinates are floating-point for sub-pixel accuracy.",
    )
    keypoints_visible: List[Union[float, List[float]]] = Field(
        ...,
        description="Visibility/confidence scores for each keypoint, range [0.0, 1.0]. "
        "Higher values indicate higher confidence that the keypoint is visible "
        "and correctly localized. Length should match number of keypoints.",
    )
    pose_bbox: List[float] = Field(
        ...,
        min_items=4,
        max_items=4,
        description="Pose-refined bounding box coordinates in pixels [x1, y1, x2, y2] format. "
        "This bbox is refined/adjusted by the pose estimation model based on keypoint locations "
        "and may be tighter or more accurate than the original detection bbox. "
        "Often computed from actual keypoint positions for better person localization. "
        "Zero-based pixel coordinates: (0,0) at top-left, x increases right, y down.",
    )
    pose_bbox_scores: List[float] = Field(
        ...,
        description="Pose estimation bounding box confidence scores from the pose model. "
        "May contain multiple confidence values: overall pose confidence, bbox quality score, "
        "keypoint coverage score, etc. Different from detection score as this reflects "
        "pose estimation quality rather than object detection confidence.",
    )
    pose_model: str = Field(
        ...,
        description="Name or identifier of the pose estimation model used to generate these keypoints. "
        "Examples: 'rtmpose-m', 'dwpose', 'alphapose', 'mediapipe', 'openpose', etc. "
        "This field is crucial for interpreting keypoint order, coordinate system, "
        "and model-specific confidence thresholds. Different models may use different "
        "keypoint schemas (e.g., COCO-17, COCO-133, MPII-16).",
    )


class Person(BaseModel):
    """
    Data for a single person tracked across multiple frames.

    Represents a unique individual tracked through a video sequence, containing
    all their detections and pose estimations across different frames.
    """

    person_id: int = Field(
        ...,
        ge=0,
        description="Unique identifier for this person within the video. "
        "Used for tracking the same individual across multiple frames. "
        "Typically starts from 0 and increments for each new person detected.",
    )
    detections: List[Detection] = Field(
        default_factory=list,
        description="List of all object detections for this person across video frames. "
        "Each detection contains bounding box and confidence information "
        "for frames where this person was detected.",
    )
    poses: List[PoseData] = Field(
        default_factory=list,
        description="List of all pose estimations for this person across video frames. "
        "Each pose contains keypoint coordinates and visibility scores "
        "for frames where pose estimation was performed.",
    )

    def add_detection(
        self, frame_idx: int, bbox: List[float], score: float, label: int = 0
    ):
        """Add a detection for this person in a specific frame."""
        detection = Detection(
            detection_frame_idx=frame_idx,
            detection_bbox=bbox,
            detection_score=score,
            object_label=label,
        )
        self.detections.append(detection)

    def add_pose(
        self,
        frame_idx: int,
        keypoints: List[List[float]],
        keypoints_visible: List[float],
        bbox: List[float],
        bbox_scores: List[float],
        pose_model: str,
    ):
        """Add pose data for this person in a specific frame."""
        pose = PoseData(
            frame_idx=frame_idx,
            keypoints=keypoints,
            keypoints_visible=keypoints_visible,
            pose_bbox=bbox,
            pose_bbox_scores=bbox_scores,
            pose_model=pose_model,
        )
        self.poses.append(pose)


class VideoData(BaseModel):
    """
    Complete video data with person tracking and pose estimation results.

    Top-level container for all detection and pose estimation data from a single video.
    Maintains person tracking across frames and stores both per-person data and
    frame-level detection summaries.
    """

    video_name: str = Field(
        ...,
        description="Name or identifier of the video file (without extension). "
        "Used for organizing results and matching with source video files.",
    )
    persons: List[Person] = Field(
        default_factory=list,
        description="List of all tracked persons in the video. Each person contains "
        "their complete detection and pose history across all frames.",
    )
    all_detections_per_frame: Dict[int, List[Detection]] = Field(
        default_factory=dict,
        description="Frame-indexed dictionary of all detections (all object classes). "
        "Key: frame_idx (int), Value: list of all detections in that frame. "
        "Includes non-human objects depending on detection model configuration.",
    )
    detection_config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Global configuration parameters used during detection/pose estimation for this entire video. "
        "Contains model settings, thresholds, device info, etc. applied to ALL detections/poses. "
        "Stored at video level to avoid duplication across individual Detection/PoseData objects. "
        "Essential for reproducibility and result interpretation across the entire video sequence.",
    )

    def add_person(self, person_id: int) -> Person:
        """Add a new person to track."""
        person = Person(person_id=person_id)
        self.persons.append(person)
        return person

    def get_person(self, person_id: int) -> Optional[Person]:
        """Get person by ID."""
        for person in self.persons:
            if person.person_id == person_id:
                return person
        return None

    def get_or_create_person(self, person_id: int) -> Person:
        """Get existing person or create new one."""
        person = self.get_person(person_id)
        if person is None:
            person = self.add_person(person_id)
        return person

    def add_frame_detections(
        self,
        frame_idx: int,
        all_bboxes: List[List[float]],
        all_scores: List[float],
        all_labels: List[int],
    ):
        """Store all detections for a frame (not just humans)."""
        frame_detections = []
        for bbox, score, label in zip(all_bboxes, all_scores, all_labels):
            detection = Detection(
                detection_frame_idx=frame_idx,
                detection_bbox=bbox,
                detection_score=score,
                object_label=label,
            )
            frame_detections.append(detection)
        self.all_detections_per_frame[frame_idx] = frame_detections

    def get_frame_detections(self, frame_idx: int) -> List[Detection]:
        """Get all detections for a specific frame."""
        return self.all_detections_per_frame.get(frame_idx, [])

    def get_human_detections_at_frame(self, frame_idx: int) -> List[Detection]:
        """Get only human detections (label == 0) for a specific frame."""
        frame_detections = self.get_frame_detections(frame_idx)
        return [det for det in frame_detections if det.object_label == 0]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "video_name": self.video_name,
            "persons": [
                {
                    "person_id": person.person_id,
                    "detections": [
                        {
                            "frame_idx": det.detection_frame_idx,
                            "bbox": det.detection_bbox,
                            "score": det.detection_score,
                            "label": det.object_label,
                        }
                        for det in person.detections
                    ],
                    "poses": [
                        {
                            "frame_idx": pose.frame_idx,
                            "keypoints": pose.keypoints,
                            "keypoints_visible": pose.keypoints_visible,
                            "bbox": pose.pose_bbox,
                            "bbox_scores": pose.pose_bbox_scores,
                            "pose_model": pose.pose_model,
                        }
                        for pose in person.poses
                    ],
                }
                for person in self.persons
            ],
            "detection_config": self.detection_config,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VideoData":
        """Create VideoData from dictionary."""
        video_data = cls(
            video_name=data["video_name"], detection_config=data.get(
                "detection_config")
        )

        # Load persons
        for person_data in data.get("persons", []):
            person = video_data.add_person(person_data["person_id"])

            # Load detections
            for det_data in person_data.get("detections", []):
                person.add_detection(
                    det_data["frame_idx"],
                    det_data["bbox"],
                    det_data["score"],
                    det_data["label"],
                )

            # Load poses
            for pose_data in person_data.get("poses", []):
                person.add_pose(
                    pose_data["frame_idx"],
                    pose_data["keypoints"],
                    pose_data["keypoints_visible"],
                    pose_data["bbox"],
                    pose_data["bbox_scores"],
                    pose_data.get(
                        "pose_model", "unknown"
                    ),  # Default for backward compatibility
                )

        # Reconstruct all_detections_per_frame from person data
        # This maintains functionality while avoiding redundant storage
        for person in video_data.persons:
            for detection in person.detections:
                frame_idx = detection.detection_frame_idx
                if frame_idx not in video_data.all_detections_per_frame:
                    video_data.all_detections_per_frame[frame_idx] = []
                video_data.all_detections_per_frame[frame_idx].append(
                    detection)

        # Load legacy all_detections_per_frame if present (for backward compatibility)
        legacy_detections = data.get("all_detections_per_frame", {})
        for frame_idx_str, detections_data in legacy_detections.items():
            frame_idx = int(frame_idx_str)
            # Only add if not already reconstructed from person data
            if frame_idx not in video_data.all_detections_per_frame:
                frame_detections = []
                for det_data in detections_data:
                    detection = Detection(
                        detection_frame_idx=det_data["frame_idx"],
                        detection_bbox=det_data["bbox"],
                        detection_score=det_data["score"],
                        object_label=det_data["label"],
                    )
                    frame_detections.append(detection)
                video_data.all_detections_per_frame[frame_idx] = frame_detections

        return video_data


def calculate_bbox_overlap(bbox1: List[float], bbox2: List[float]) -> float:
    """Calculate IoU (Intersection over Union) between two bounding boxes."""
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)

    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0

    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def calculate_bbox_distance(bbox1: List[float], bbox2: List[float]) -> float:
    """Calculate center distance between two bounding boxes."""
    center1_x = (bbox1[0] + bbox1[2]) / 2
    center1_y = (bbox1[1] + bbox1[3]) / 2
    center2_x = (bbox2[0] + bbox2[2]) / 2
    center2_y = (bbox2[1] + bbox2[3]) / 2

    return np.sqrt((center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2)

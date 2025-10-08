from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import numpy as np


@dataclass
class Detection:
    """Single detection data for a frame."""

    frame_idx: int
    bbox: List[float]  # [x1, y1, x2, y2]
    score: float
    label: int


@dataclass
class PoseData:
    """Pose estimation data for a single frame."""

    frame_idx: int
    keypoints: List[List[float]]  # (J, 2) - x, y coordinates
    keypoints_visible: List[float]  # Visibility scores for each keypoint
    bbox: List[float]  # [x1, y1, x2, y2] - bbox from pose estimation
    bbox_scores: List[float]  # Bbox confidence scores from pose estimation


@dataclass
class Person:
    """Data for a single person tracked across multiple frames."""

    person_id: int
    detections: List[Detection] = field(default_factory=list)
    poses: List[PoseData] = field(default_factory=list)

    def add_detection(
        self, frame_idx: int, bbox: List[float], score: float, label: int = 0
    ):
        """Add a detection for this person in a specific frame."""
        detection = Detection(frame_idx=frame_idx, bbox=bbox, score=score, label=label)
        self.detections.append(detection)

    def add_pose(
        self,
        frame_idx: int,
        keypoints: List[List[float]],
        keypoints_visible: List[float],
        bbox: List[float],
        bbox_scores: List[float],
    ):
        """Add pose data for this person in a specific frame."""
        pose = PoseData(
            frame_idx=frame_idx,
            keypoints=keypoints,
            keypoints_visible=keypoints_visible,
            bbox=bbox,
            bbox_scores=bbox_scores,
        )
        self.poses.append(pose)


@dataclass
class VideoData:
    """Complete video data with person tracking."""

    video_name: str
    persons: List[Person] = field(default_factory=list)
    all_detections_per_frame: Dict[int, List[Detection]] = field(default_factory=dict)
    detection_config: Optional[Dict[str, Any]] = None

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
                frame_idx=frame_idx, bbox=bbox, score=score, label=label
            )
            frame_detections.append(detection)
        self.all_detections_per_frame[frame_idx] = frame_detections

    def get_frame_detections(self, frame_idx: int) -> List[Detection]:
        """Get all detections for a specific frame."""
        return self.all_detections_per_frame.get(frame_idx, [])

    def get_human_detections_at_frame(self, frame_idx: int) -> List[Detection]:
        """Get only human detections (label == 0) for a specific frame."""
        frame_detections = self.get_frame_detections(frame_idx)
        return [det for det in frame_detections if det.label == 0]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "video_name": self.video_name,
            "persons": [
                {
                    "person_id": person.person_id,
                    "detections": [
                        {
                            "frame_idx": det.frame_idx,
                            "bbox": det.bbox,
                            "score": det.score,
                            "label": det.label,
                        }
                        for det in person.detections
                    ],
                    "poses": [
                        {
                            "frame_idx": pose.frame_idx,
                            "keypoints": pose.keypoints,
                            "keypoints_visible": pose.keypoints_visible,
                            "bbox": pose.bbox,
                            "bbox_scores": pose.bbox_scores,
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
            video_name=data["video_name"], detection_config=data.get("detection_config")
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
                )

        # Reconstruct all_detections_per_frame from person data
        # This maintains functionality while avoiding redundant storage
        for person in video_data.persons:
            for detection in person.detections:
                frame_idx = detection.frame_idx
                if frame_idx not in video_data.all_detections_per_frame:
                    video_data.all_detections_per_frame[frame_idx] = []
                video_data.all_detections_per_frame[frame_idx].append(detection)

        # Load legacy all_detections_per_frame if present (for backward compatibility)
        legacy_detections = data.get("all_detections_per_frame", {})
        for frame_idx_str, detections_data in legacy_detections.items():
            frame_idx = int(frame_idx_str)
            # Only add if not already reconstructed from person data
            if frame_idx not in video_data.all_detections_per_frame:
                frame_detections = []
                for det_data in detections_data:
                    detection = Detection(
                        frame_idx=det_data["frame_idx"],
                        bbox=det_data["bbox"],
                        score=det_data["score"],
                        label=det_data["label"],
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

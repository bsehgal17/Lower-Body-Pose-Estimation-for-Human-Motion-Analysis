from typing import List
from utils.data_structures import (
    VideoData,
    calculate_bbox_overlap,
    calculate_bbox_distance,
)


class PersonTracker:
    """Simple person tracker using bounding box overlap and distance."""

    def __init__(
        self, overlap_threshold: float = 0.3, distance_threshold: float = 100.0
    ):
        """
        Initialize person tracker.

        Args:
            overlap_threshold: Minimum IoU to consider same person
            distance_threshold: Maximum center distance to consider same person
        """
        self.overlap_threshold = overlap_threshold
        self.distance_threshold = distance_threshold
        self.next_person_id = 0

    def assign_person_ids(
        self,
        video_data: VideoData,
        human_bboxes_per_frame: List[List[List[float]]],
        human_scores_per_frame: List[List[float]],
        frame_indices: List[int],
    ) -> List[List[int]]:
        """
        Assign person IDs to human detections across frames.

        Args:
            video_data: VideoData object to store results
            human_bboxes_per_frame: List of human bboxes for each frame
            human_scores_per_frame: List of human scores for each frame
            frame_indices: List of frame indices

        Returns:
            List of person IDs for each detection in each frame
        """
        person_ids_per_frame = []
        active_persons = []  # List of (person_id, last_bbox, last_frame_idx)

        for frame_idx, (frame_bboxes, frame_scores) in zip(
            frame_indices, zip(human_bboxes_per_frame, human_scores_per_frame)
        ):
            frame_person_ids = []
            current_frame_persons = []

            for bbox, score in zip(frame_bboxes, frame_scores):
                # Try to match with existing persons
                best_match = None
                best_match_score = 0.0

                for i, (person_id, last_bbox, last_frame_idx) in enumerate(
                    active_persons
                ):
                    # Skip if too many frames have passed
                    if frame_idx - last_frame_idx > 5:  # Max gap of 5 frames
                        continue

                    # Calculate overlap and distance
                    overlap = calculate_bbox_overlap(bbox, last_bbox)
                    distance = calculate_bbox_distance(bbox, last_bbox)

                    # Score based on overlap and distance
                    if (
                        overlap >= self.overlap_threshold
                        and distance <= self.distance_threshold
                    ):
                        match_score = overlap * (
                            1.0 - distance / self.distance_threshold
                        )
                        if match_score > best_match_score:
                            best_match = i
                            best_match_score = match_score

                if best_match is not None:
                    # Use existing person
                    person_id, _, _ = active_persons[best_match]
                    frame_person_ids.append(person_id)
                    current_frame_persons.append((person_id, bbox, frame_idx))

                    # Remove from active_persons (will be re-added with new position)
                    active_persons.pop(best_match)
                else:
                    # Create new person
                    person_id = self.next_person_id
                    self.next_person_id += 1
                    frame_person_ids.append(person_id)
                    current_frame_persons.append((person_id, bbox, frame_idx))

                    # Add person to video data
                    video_data.add_person(person_id)

            # Update active persons with current frame data
            active_persons.extend(current_frame_persons)

            # Remove persons that haven't been seen for too long
            active_persons = [
                (pid, bbox, fid)
                for pid, bbox, fid in active_persons
                if frame_idx - fid <= 5
            ]

            person_ids_per_frame.append(frame_person_ids)

        return person_ids_per_frame

    def track_and_store(
        self,
        video_data: VideoData,
        frame_idx: int,
        all_bboxes: List[List[float]],
        all_scores: List[float],
        all_labels: List[int],
        human_bboxes: List[List[float]],
        human_scores: List[float],
        pose_results: List,
        person_ids: List[int],
    ):
        """
        Store detection and pose data with person tracking.

        Args:
            video_data: VideoData object to store results
            frame_idx: Current frame index
            all_bboxes: All detected bounding boxes
            all_scores: All detection scores
            all_labels: All detection labels
            human_bboxes: Human-only bounding boxes
            human_scores: Human-only scores
            pose_results: Pose estimation results
            person_ids: Assigned person IDs for each human detection
        """
        # Store all detections for this frame
        video_data.add_frame_detections(frame_idx, all_bboxes, all_scores, all_labels)

        # Store human detections and poses per person
        for i, (bbox, score, person_id) in enumerate(
            zip(human_bboxes, human_scores, person_ids)
        ):
            person = video_data.get_or_create_person(person_id)

            # Add detection
            person.add_detection(frame_idx, bbox, score, label=0)  # 0 for human

            # Add pose if available
            if i < len(pose_results):
                pose_result = pose_results[i]
                keypoints = pose_result.pred_instances.keypoints.tolist()
                keypoints_visible = (
                    pose_result.pred_instances.keypoints_visible.tolist()
                )
                bbox_scores = pose_result.pred_instances.bbox_scores.tolist()
                person.add_pose(
                    frame_idx, keypoints, keypoints_visible, bbox, bbox_scores
                )

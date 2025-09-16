import cv2
from utils.data_structures import VideoData
from utils.person_tracker import PersonTracker


class FrameProcessor:
    def __init__(self, detector, estimator, visualizer, config):
        self.detector = detector
        self.estimator = estimator
        self.visualizer = visualizer
        self.config = config
        self.person_tracker = PersonTracker()
        self.frame_buffer = []  # Store frame data for batch person tracking

    def process_frame(self, frame, frame_idx, video_data: VideoData):
        """Process single frame and store detection data for each person separately."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        all_bboxes, all_scores, all_labels = self.detector.detect_humans(rgb)

        # Filter for human detections (label == 0) with confidence threshold for pose estimation
        human_indices = [
            i
            for i, (score, label) in enumerate(zip(all_scores, all_labels))
            if label == 0 and score > self.config.processing.detection_threshold
        ]

        if human_indices:
            human_bboxes = [all_bboxes[i] for i in human_indices]
            human_scores = [all_scores[i] for i in human_indices]

            data_samples, pose_results = self.estimator.estimate_pose(
                frame, human_bboxes
            )

            # Store each person's data separately with temporary person IDs
            for person_idx, (bbox, score, pose_result) in enumerate(
                zip(human_bboxes, human_scores, pose_results)
            ):
                # Use frame-based person ID for now (will be properly tracked later)
                temp_person_id = f"frame_{frame_idx}_person_{person_idx}"

                self.frame_buffer.append(
                    {
                        "frame_idx": frame_idx,
                        "temp_person_id": temp_person_id,
                        "bbox": bbox,
                        "score": score,
                        "pose_result": pose_result,
                        "frame": frame,
                    }
                )

            return self.visualizer.visualize_pose(frame, data_samples)
        else:
            # No human detections - still need to store frame info
            return frame

    def finalize_video_processing(self, video_data: VideoData):
        """Process all frames with person tracking after video is complete."""
        if not self.frame_buffer:
            return

        # Group detections by frame for person tracking
        frames_data = {}
        for item in self.frame_buffer:
            frame_idx = item["frame_idx"]
            if frame_idx not in frames_data:
                frames_data[frame_idx] = []
            frames_data[frame_idx].append(item)

        # Sort frames by index
        sorted_frame_indices = sorted(frames_data.keys())

        # Extract data for person tracking
        human_bboxes_per_frame = []
        human_scores_per_frame = []

        for frame_idx in sorted_frame_indices:
            frame_detections = frames_data[frame_idx]
            frame_bboxes = [item["bbox"] for item in frame_detections]
            frame_scores = [item["score"] for item in frame_detections]
            human_bboxes_per_frame.append(frame_bboxes)
            human_scores_per_frame.append(frame_scores)

        # Assign person IDs across all frames
        person_ids_per_frame = self.person_tracker.assign_person_ids(
            video_data,
            human_bboxes_per_frame,
            human_scores_per_frame,
            sorted_frame_indices,
        )

        # Store each person's data separately
        for frame_idx, frame_person_ids in zip(
            sorted_frame_indices, person_ids_per_frame
        ):
            frame_detections = frames_data[frame_idx]

            for detection_item, person_id in zip(frame_detections, frame_person_ids):
                # Get or create person
                person = video_data.get_or_create_person(person_id)

                # Add detection data
                person.add_detection(
                    frame_idx, detection_item["bbox"], detection_item["score"], label=0
                )

                # Add pose data
                pose_result = detection_item["pose_result"]
                keypoints = pose_result.pred_instances.keypoints.tolist()
                keypoints_visible = (
                    pose_result.pred_instances.keypoints_visible.tolist()
                )
                bboxes = pose_result.pred_instances.bboxes.tolist()
                bbox_scores = pose_result.pred_instances.bbox_scores.tolist()
                person.add_pose(
                    frame_idx, keypoints, keypoints_visible, bboxes, bbox_scores
                )

        # Clear buffer
        self.frame_buffer.clear()

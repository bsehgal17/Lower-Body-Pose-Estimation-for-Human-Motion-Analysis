import pickle
import numpy as np
import os
import json
from utils.data_structures import VideoData


def save_video_data_to_json(video_data: VideoData, save_dir: str, video_name: str):
    """Save VideoData to JSON file."""
    output_json_path = os.path.join(save_dir, f"{video_name}.json")

    # Convert to dictionary and save
    output_dict = video_data.to_dict()

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(output_dict, f, indent=2)


def load_video_data_from_json(json_path: str) -> VideoData:
    """Load VideoData from JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        data_dict = json.load(f)

    return VideoData.from_dict(data_dict)


def combine_keypoints(pose_results, frame_idx, video_data, detection_data):
    """Legacy function - kept for backward compatibility but should not be used with new structure."""
    # This function is deprecated when using VideoData with person tracking
    # The new structure handles data storage through the FrameProcessor.finalize_video_processing
    pass


def save_keypoints_to_json(
    video_data, save_dir, video_name, detector_config: dict = None
):
    """Legacy function - updated to work with new VideoData structure."""
    if isinstance(video_data, VideoData):
        # New structure
        if detector_config:
            video_data.detection_config = detector_config
        save_video_data_to_json(video_data, save_dir, video_name)
    else:
        # Legacy structure - convert to new format first
        legacy_video_data = VideoData(
            video_name=video_name, detection_config=detector_config
        )

        # Convert legacy format to new format (basic conversion)
        for frame_data in video_data:
            frame_idx = frame_data["frame_idx"]

            # Extract detection data if available
            if "detection_data" in frame_data:
                det_data = frame_data["detection_data"]
                legacy_video_data.add_frame_detections(
                    frame_idx,
                    det_data["all_bboxes"],
                    det_data["all_scores"],
                    det_data["all_labels"],
                )

            # Convert keypoints to person data (simple assignment without tracking)
            for i, person_data in enumerate(frame_data.get("keypoints", [])):
                person_id = i  # Simple assignment - not optimal but functional
                person = legacy_video_data.get_or_create_person(person_id)

                if "keypoints" in person_data:
                    keypoints = person_data["keypoints"]
                    keypoint_scores = person_data.get("scores", [])
                    bbox = person_data.get("bboxes", [])

                    if bbox:
                        person.add_detection(frame_idx, bbox, 1.0, label=0)
                        person.add_pose(frame_idx, keypoints, keypoint_scores, bbox)

        save_video_data_to_json(legacy_video_data, save_dir, video_name)


def unpack_prediction_pkl(pkl_path, person_idx=0):
    """
    Unpacks prediction data. Updated to work with new VideoData structure.

    Args:
        pkl_path (str): Path to the saved .pkl or .json file.
        person_idx (int): Index of the person per video (default: 0).

    Returns:
        np.ndarray: Array of shape (N, J, 2) for keypoints
    """
    if pkl_path.endswith(".json"):
        # New format
        video_data = load_video_data_from_json(pkl_path)

        if person_idx >= len(video_data.persons):
            raise ValueError(f"Video has fewer than {person_idx + 1} persons.")

        person = video_data.persons[person_idx]
        poses = sorted(person.poses, key=lambda x: x.frame_idx)

        keypoints_list = []
        for pose in poses:
            # Convert (J, 3) to (J, 2) by taking only x, y coordinates
            kpts = np.array(pose.keypoints)[:, :2]  # Remove confidence/visibility
            keypoints_list.append(kpts)

        return np.stack(keypoints_list, axis=0) if keypoints_list else np.array([])

    else:
        # Legacy .pkl format
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        keypoints_list = []
        for frame_data in data["keypoints"]:
            people = frame_data["keypoints"]
            if len(people) <= person_idx:
                raise ValueError(
                    f"Frame {frame_data['frame_idx']} has fewer than {person_idx + 1} people."
                )

            kpts = np.array(people[person_idx]["keypoints"])  # (J, 2)
            keypoints_list.append(kpts)

        return np.stack(keypoints_list, axis=0)  # (N, J, 2)

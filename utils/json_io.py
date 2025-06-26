import pickle
import numpy as np
import os
import json


def combine_keypoints(pose_results, frame_idx, video_data, bboxes):
    """Appends keypoints and predictions for each frame to the video_data list."""
    frame_data = {
        "frame_idx": frame_idx,
        "keypoints": [
            {
                "keypoints": person.pred_instances.keypoints.tolist(),
                "scores": person.pred_instances.keypoint_scores.tolist(),
                "bboxes": bboxes.tolist(),
            }
            for person in pose_results
        ],
    }
    video_data.append(frame_data)


def save_keypoints_to_json(video_data, save_dir, video_name, detector_config: dict = None):
    output_json_path = os.path.join(save_dir, f"{video_name}.json")

    # Bundle keypoints and detector config
    output = {
        "keypoints": video_data,
    }

    if detector_config:
        output["detection_config"] = detector_config

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)


def unpack_prediction_pkl(pkl_path, person_idx=0):
    """
    Unpacks your nested prediction .pkl into (N, J, 2) format.

    Args:
        pkl_path (str): Path to the saved .pkl file.
        person_idx (int): Index of the person per frame (default: 0).

    Returns:
        np.ndarray: Array of shape (N, J, 2)
    """
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    keypoints_list = []
    for frame_data in data["keypoints"]:
        people = frame_data["keypoints"]
        if len(people) <= person_idx:
            raise ValueError(
                f"Frame {frame_data['frame_idx']} has fewer than {person_idx+1} people.")

        kpts = np.array(people[person_idx]["keypoints"])  # (J, 2)
        keypoints_list.append(kpts)

    return np.stack(keypoints_list, axis=0)  # (N, J, 2)

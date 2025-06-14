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

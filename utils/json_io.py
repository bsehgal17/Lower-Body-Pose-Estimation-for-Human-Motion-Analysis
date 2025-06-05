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


def save_keypoints_to_json(video_data, output_dir, video_name):
    """Saves all keypoints and predictions for the entire video in a single JSON file."""
    video_output_dir = os.path.join(output_dir, os.path.splitext(video_name)[0])
    os.makedirs(video_output_dir, exist_ok=True)

    output_file = os.path.join(
        video_output_dir, f"{os.path.splitext(video_name)[0]}.json"
    )
    with open(output_file, "w") as f:
        json.dump(video_data, f, indent=4)
    print(f"Saved keypoints and predictions for the entire video to {output_file}")

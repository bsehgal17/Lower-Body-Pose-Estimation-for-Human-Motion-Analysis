from typing import Dict, Optional, Tuple
from typing import List, Union
import os
import json
import cv2
import importlib


def visualize_selected_joints(
    video_path: str,
    json_file: str,
    output_video_path: str,
    joint_specs: Optional[List[Tuple[str, str]]] = None,
    frame_range: Optional[Union[List[int], Tuple[int, int]]] = None,
):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"JSON not found: {json_file}")

    output_dir = os.path.dirname(output_video_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(json_file, "r") as f:
        predictions = json.load(f)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Error: Couldn't open the video.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_range is None:
        frame_range = (0, total_frames)
    start_frame, end_frame = frame_range
    end_frame = min(end_frame, total_frames)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_size = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    out = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

    # Skeleton connections for lower body
    skeleton_connections = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]

    joint_indices = None
    if joint_specs:
        joint_indices = []
        try:
            module = importlib.import_module("utils.joint_enum")
            for enum_name, joint_name in joint_specs:
                enum_cls = getattr(module, enum_name)
                joint_indices.append(enum_cls[joint_name].value)
        except Exception as e:
            raise ValueError(f"Invalid joint spec in {joint_specs}: {e}")

    for frame_num in range(start_frame, end_frame):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Couldn't read frame {frame_num}.")
            continue

        frame_data = next(
            (item for item in predictions if item["frame_idx"] == frame_num), None
        )
        if frame_data is None:
            print(f"Warning: No predictions for frame {frame_num}")
            continue

        for keypoint_group in frame_data["keypoints"]:
            for keypoint_set in keypoint_group["keypoints"]:
                for idx, (x, y) in enumerate(keypoint_set):
                    if joint_indices and idx not in joint_indices:
                        continue
                    cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

                for start_idx, end_idx in skeleton_connections:
                    if start_idx < len(keypoint_set) and end_idx < len(keypoint_set):
                        x1, y1 = keypoint_set[start_idx]
                        x2, y2 = keypoint_set[end_idx]
                        cv2.line(
                            frame,
                            (int(x1), int(y1)),
                            (int(x2), int(y2)),
                            (0, 255, 0),
                            1,
                        )

        out.write(frame)

    cap.release()
    out.release()
    print(f"Output video saved to: {output_video_path}")

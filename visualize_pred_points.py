import cv2
import json
import numpy as np
import os


def visualize_predictions(
    video_path, json_file, output_video_path, frame_range=None, joint_indices=None
):
    """
    Visualizes predicted keypoints from a JSON file on video frames with correct positioning.

    Parameters:
    - video_path (str): Path to the video file.
    - json_file (str): Path to the JSON file containing predicted keypoints.
    - output_video_path (str): Path to save the output video with overlaid keypoints.
    - frame_range (tuple, optional): Start and end frame indices. If None, all frames are processed.
    - joint_indices (list, optional): List of joint indices to display. If None, all joints are displayed.

    Returns:
    - None (Saves a video file with overlaid keypoints).
    """
    # Ensure output directory exists
    output_dir = os.path.dirname(output_video_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load JSON file
    with open(json_file, "r") as f:
        predictions = json.load(f)

    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Couldn't open the video.")
        return

    # Get total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_range is None:
        frame_range = (0, total_frames)

    start_frame, end_frame = frame_range
    end_frame = min(end_frame, total_frames)

    # Get the video codec and create VideoWriter to save the output video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_size = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    out = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

    # Define skeleton connections for visualization
    skeleton_connections = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]

    for frame_num in range(start_frame, end_frame):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Couldn't read frame {frame_num}.")
            continue

        # Extract keypoints for the current frame
        frame_data = next(
            (item for item in predictions if item["frame_idx"] == frame_num), None
        )
        if frame_data is None:
            print(f"Warning: No predictions available for frame {frame_num}")
            continue

        for keypoint_group in frame_data["keypoints"]:
            for keypoint_set in keypoint_group["keypoints"]:
                # Draw keypoints
                for idx, (x, y) in enumerate(keypoint_set):
                    if joint_indices and idx not in joint_indices:
                        continue
                    cv2.circle(
                        frame, (int(x), int(y)), 5, (0, 255, 0), -1
                    )  # Green circle

                # Draw skeleton connections
                for start_idx, end_idx in skeleton_connections:
                    if start_idx < len(keypoint_set) and end_idx < len(keypoint_set):
                        x_start, y_start = keypoint_set[start_idx]
                        x_end, y_end = keypoint_set[end_idx]
                        cv2.line(
                            frame,
                            (int(x_start), int(y_start)),
                            (int(x_end), int(y_end)),
                            (0, 255, 0),
                            1,
                        )  # Green line

        # Write the frame with overlaid keypoints to the output video
        out.write(frame)

    cap.release()
    out.release()
    print(f"Output video saved to: {output_video_path}")

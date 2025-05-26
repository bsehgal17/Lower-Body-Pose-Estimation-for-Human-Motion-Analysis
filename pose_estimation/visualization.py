import mmcv
import os
import cv2
from mmpose.registry import VISUALIZERS
from utils.config import OUTPUT_DIR
from tqdm import tqdm


def init_visualizer(pose_estimator):
    """Initialize visualizer."""
    pose_estimator.cfg.visualizer.radius = 3
    pose_estimator.cfg.visualizer.line_width = 1
    visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)
    visualizer.set_dataset_meta(pose_estimator.dataset_meta)
    return visualizer


def create_video_from_frames(frames, video_output_path, fps=60):
    """Creates a video from the given frames."""
    if len(frames) == 0:
        print("No frames to create video.")
        return

    # Get the frame size (width, height) from the first frame
    height, width, _ = frames[0].shape
    # You can change this to 'XVID' or other formats
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(
        video_output_path, fourcc, fps, (width, height))

    # Write all frames to the video file
    for frame in frames:
        video_writer.write(frame)

    video_writer.release()
    print(f"Video saved to {video_output_path}")


def visualize_pose(frame, data_samples, visualizer, frame_idx, frames_list):
    """Visualizes pose estimation results, adds them to frames list, and later saves as video."""
    visualizer.add_datasample(
        "result",
        frame,
        data_sample=data_samples,
        draw_gt=False,
        draw_heatmap=False,
        draw_bbox=False,
        show=False,
        wait_time=0,
        out_file=None,
        kpt_thr=0.3,
    )

    vis_result = visualizer.get_image()

    # Append the visualized frame to the list of frames
    frames_list.append(vis_result)


def visualize_lower_points(frame, data_samples, frame_idx, frames_list, joints_to_visualize):
    """
    Visualizes specific pose estimation results based on given joint indices,
    adds them to the frames list, and later saves as video.

    :param frame: The current frame to visualize.
    :param data_samples: The pose estimation result for the current frame (single PoseDataSample object).
    :param frame_idx: The current frame index.
    :param frames_list: List to store the frames with visualized poses.
    :param joints_to_visualize: A list of joint indices to visualize (e.g., [0, 1, 2] for hip, knee, ankle).
    :param threshold: Minimum confidence to consider keypoints (0 to 1).
    """
    # Extract keypoints from data_samples
    # Assuming this is an array of shape (1, num_joints, 3)
    keypoints = data_samples.pred_instances.keypoints

    # Prepare the frame copy to draw keypoints on
    frame_copy = frame.copy()

    # Iterate over the joints to visualize
    for joint_idx in joints_to_visualize:
        if joint_idx < len(keypoints[0]):
            # Extract (x, y, confidence) for each keypoint
            x, y = keypoints[0][joint_idx]

            cv2.circle(frame_copy, (int(x), int(y)), 2,
                       (0, 255, 0), -1)  # Green color

    # Optionally, you can draw skeleton by connecting joints (example for a couple of joints)
    # Connect joints using lines (e.g., hip -> knee -> ankle)
    skeleton_connections = [
        (11, 12),
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16)
    ]

    for start_idx, end_idx in skeleton_connections:
        if start_idx < len(keypoints[0]) and end_idx < len(keypoints[0]):
            x_start, y_start = keypoints[0][start_idx]
            x_end, y_end, = keypoints[0][end_idx]

            # Draw lines between joints if both have sufficient confidence
            cv2.line(frame_copy, (int(x_start), int(y_start)),
                     (int(x_end), int(y_end)), (0, 255, 0), 1)  # Green line

    # Add the frame with keypoints to the frames list
    frames_list.append(frame_copy)

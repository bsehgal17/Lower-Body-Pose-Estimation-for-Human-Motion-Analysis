import mmcv
import os
import cv2
from mmpose.registry import VISUALIZERS
from config import OUTPUT_DIR
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

import mmcv
import os
from mmpose.registry import VISUALIZERS
from config import OUTPUT_DIR

def init_visualizer(pose_estimator):
    """Initialize visualizer."""
    pose_estimator.cfg.visualizer.radius = 3
    pose_estimator.cfg.visualizer.line_width = 1
    visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)
    visualizer.set_dataset_meta(pose_estimator.dataset_meta)
    return visualizer


def visualize_pose(frame, data_samples, visualizer, frame_idx):
    """Visualizes pose estimation results and saves output images."""
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
    output_image_file = os.path.join(OUTPUT_DIR, f"frame_{frame_idx:04d}.png")
    mmcv.imwrite(vis_result, output_image_file)
    print(f"Saved visualization of frame {frame_idx} to {output_image_file}")

import mmcv
import os
import cv2
import numpy as np
from mmpose.registry import VISUALIZERS
from config.pipeline_config import PipelineConfig


class PoseVisualizer:
    def __init__(self, pose_estimator, config: PipelineConfig):
        self.config = config

        # Set visualizer parameters
        pose_estimator.cfg.visualizer.radius = 3
        pose_estimator.cfg.visualizer.line_width = 1
        self.visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)

        # Use full meta but truncate later for display
        self.visualizer.set_dataset_meta(pose_estimator.dataset_meta)

    def visualize_pose(self, frame, data_samples):
        """Visualizes only body + feet (joints 0–24) by modifying data on the fly."""
        data_samples = data_samples.clone() if hasattr(
            data_samples, 'clone') else data_samples

        # Slice keypoints and scores
        data_samples.pred_instances.keypoints[0][25:] = np.array(
            [[0.0, 0.0]] * (data_samples.pred_instances.keypoints.shape[1] - 25))
        if hasattr(data_samples.pred_instances, 'keypoint_scores'):
            data_samples.pred_instances.keypoint_scores[0][25:] = 0.0

        self.visualizer.add_datasample(
            "result",
            frame,
            data_sample=data_samples,
            draw_gt=False,
            draw_heatmap=False,
            draw_bbox=False,
            show=False,
            wait_time=0,
            out_file=None,
            kpt_thr=self.config.processing.kpt_threshold,
        )
        return self.visualizer.get_image()

    def visualize_lower_points(self, frame, data_samples, frame_idx, frames_list, joints_to_visualize):
        """Visualizes specific joints (e.g., lower body)."""
        keypoints = data_samples.pred_instances.keypoints
        frame_copy = frame.copy()

        for joint_idx in joints_to_visualize:
            if joint_idx < len(keypoints[0]):
                x, y = keypoints[0][joint_idx]
                cv2.circle(frame_copy, (int(x), int(y)), 2, (0, 255, 0), -1)

        skeleton_connections = [
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
            (19, 17), (20, 18), (17, 18),
            (23, 21), (24, 22), (21, 22),
        ]
        for start_idx, end_idx in skeleton_connections:
            if start_idx < len(keypoints[0]) and end_idx < len(keypoints[0]):
                x_start, y_start = keypoints[0][start_idx]
                x_end, y_end = keypoints[0][end_idx]
                cv2.line(
                    frame_copy,
                    (int(x_start), int(y_start)),
                    (int(x_end), int(y_end)),
                    (0, 255, 0),
                    1,
                )

        frames_list.append(frame_copy)
        return frames_list

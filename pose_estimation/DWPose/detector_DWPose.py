from pathlib import Path
import torch
import numpy as np
from mmdet.apis import inference_detector, init_detector
from mmpose.apis import inference_topdown, init_model as init_pose_model


class DWposeDetector:
    def __init__(self, pipeline_config):
        self.device = pipeline_config.processing.device
        self.max_detections = 10
        self.pipeline_config = pipeline_config

        # Initialize detection model using paths from YAML
        self.detector = init_detector(
            pipeline_config.models.det_config,
            pipeline_config.models.det_checkpoint,
            device=self.device
        )

        # Initialize pose estimation model using paths from YAML
        self.pose_estimator = init_pose_model(
            pipeline_config.models.pose_config,
            pipeline_config.models.pose_checkpoint,
            device=self.device
        )

    def detect_and_estimate(self, frame):
        # Run human detection
        det_results = inference_detector(self.detector, frame)

        # Filter out non-person detections (COCO class 0 is person)
        pred_instances = det_results.pred_instances[
            det_results.pred_instances.labels == 0
        ]
        bboxes = pred_instances.bboxes.cpu().numpy()
        scores = pred_instances.scores.cpu().numpy()

        # Filter by score threshold from YAML config
        keep = scores > self.pipeline_config.processing.detection_threshold
        bboxes = bboxes[keep][:self.max_detections]
        scores = scores[keep][:self.max_detections]

        if len(bboxes) == 0:
            return []

        # Convert to xywh format expected by MMpose
        bboxes_xywh = np.zeros_like(bboxes)
        bboxes_xywh[:, 0] = bboxes[:, 0]  # x
        bboxes_xywh[:, 1] = bboxes[:, 1]  # y
        bboxes_xywh[:, 2] = bboxes[:, 2] - bboxes[:, 0]  # w
        bboxes_xywh[:, 3] = bboxes[:, 3] - bboxes[:, 1]  # h

        # Run pose estimation
        pose_results = inference_topdown(
            self.pose_estimator,
            frame,
            bboxes_xywh,
            bbox_format='xywh'
        )

        # Format results: list of (N, J, 3) where J is number of joints
        results = [
            pred.pred_instances.keypoints for pred in pose_results
        ]

        return results

from mmengine.registry import init_default_scope
from mmdet.apis import inference_detector, init_detector
from mmdet.utils.setup_env import register_all_modules

import numpy as np
from utils.config import (
    DET_CONFIG,
    DET_CHECKPOINT,
    DEVICE,
    DETECTION_THRESHOLD,
    NMS_THRESHOLD,
)


def init_object_detector():
    """Initialize object detector."""
    detector = init_detector(DET_CONFIG, DET_CHECKPOINT, device=DEVICE)
    scope = detector.cfg.get("default_scope", "mmdet")
    if scope is not None:
        init_default_scope(scope)
    return detector


def detect_humans(detector, frame):
    """Runs object detection and filters human bounding boxes."""
    register_all_modules(True)
    detect_result = inference_detector(detector, frame)
    pred_instance = detect_result.pred_instances.cpu().numpy()

    # Extract bounding boxes with high confidence
    bboxes = np.concatenate(
        (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1
    )
    bboxes = bboxes[
        (pred_instance.labels == 0) & (pred_instance.scores > DETECTION_THRESHOLD)
    ]

    # Apply Non-Maximum Suppression (NMS)
    from mmpose.evaluation.functional import nms

    return bboxes[nms(bboxes, NMS_THRESHOLD)][:, :4]

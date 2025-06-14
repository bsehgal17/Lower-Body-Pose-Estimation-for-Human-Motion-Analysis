from mmengine.registry import init_default_scope
from mmdet.apis import inference_detector, init_detector
from mmdet.utils.setup_env import register_all_modules
import numpy as np
from config.pipeline_config import GlobalConfig


class Detector:
    def __init__(self, config: GlobalConfig = None):
        self.config = config or GlobalConfig()
        self.detector = init_detector(
            self.config.models.det_config,
            self.config.models.det_checkpoint,
            device=self.config.processing.device,
        )
        scope = self.detector.cfg.get("default_scope", "mmdet")
        if scope is not None:
            init_default_scope(scope)

    def detect_humans(self, frame):
        """Runs object detection and filters human bounding boxes."""
        register_all_modules(True)
        detect_result = inference_detector(self.detector, frame)
        pred_instance = detect_result.pred_instances.cpu().numpy()

        # Extract bounding boxes with high confidence
        bboxes = np.concatenate(
            (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1
        )
        bboxes = bboxes[
            (pred_instance.labels == 0)
            & (pred_instance.scores > self.config.processing.detection_threshold)
        ]

        # Apply Non-Maximum Suppression (NMS)
        from mmpose.evaluation.functional import nms

        return bboxes[nms(bboxes, self.config.processing.nms_threshold)][:, :4]

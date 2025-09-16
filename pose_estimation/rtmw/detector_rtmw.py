from mmengine.registry import init_default_scope
from mmdet.apis import inference_detector, init_detector
from mmdet.utils.setup_env import register_all_modules
from config.pipeline_config import PipelineConfig


class Detector:
    def __init__(self, config: PipelineConfig):
        self.config = config

        self.detector = init_detector(
            config.models.det_config,
            config.models.det_checkpoint,
            device=config.processing.device,
        )

        scope = self.detector.cfg.get("default_scope", "mmdet")
        if scope is not None:
            init_default_scope(scope)

    def detect_humans(self, frame):
        """Runs object detection and returns separate lists for bboxes, scores, and labels."""
        register_all_modules(True)
        detect_result = inference_detector(self.detector, frame)
        pred_instance = detect_result.pred_instances.cpu().numpy()

        # Get all detections without any filtering
        all_bboxes = pred_instance.bboxes
        all_scores = pred_instance.scores
        all_labels = pred_instance.labels

        # Return as separate lists
        return all_bboxes.tolist(), all_scores.tolist(), all_labels.tolist()

from mmpose.apis import inference_topdown, init_model
from mmpose.structures import merge_data_samples
from config.pipeline_config import GlobalConfig


class PoseEstimator:
    def __init__(self, config: GlobalConfig):
        self.config = config
        self.pose_estimator = init_model(
            config.models.pose_config,
            config.models.pose_checkpoint,
            device=config.processing.device,
            cfg_options=dict(model=dict(test_cfg=dict(output_heatmaps=True))),
        )
        self.cfg = self.pose_estimator.cfg
        self.dataset_meta = getattr(self.pose_estimator, "dataset_meta", None)

    def estimate_pose(self, frame, bboxes):
        results = inference_topdown(self.pose_estimator, frame, bboxes)
        return merge_data_samples(results), results

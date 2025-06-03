from mmpose.apis import inference_topdown, init_model
from mmpose.structures import merge_data_samples
from utils.config import Config


class PoseEstimator:
    def __init__(self, config: Config = None):
        self.config = config or Config()
        cfg_options = dict(model=dict(test_cfg=dict(output_heatmaps=True)))
        self.pose_estimator = init_model(
            self.config.models.POSE_CONFIG,
            self.config.models.POSE_CHECKPOINT,
            device=self.config.processing.DEVICE,
            cfg_options=cfg_options,
        )

    def estimate_pose(self, frame, bboxes):
        """Runs pose estimation and returns keypoint results."""
        pose_results = inference_topdown(self.pose_estimator, frame, bboxes)
        return merge_data_samples(pose_results), pose_results

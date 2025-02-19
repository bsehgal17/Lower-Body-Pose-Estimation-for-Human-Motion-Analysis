from mmpose.apis import inference_topdown, init_model
from mmpose.structures import merge_data_samples
from config import POSE_CONFIG, POSE_CHECKPOINT, DEVICE

def init_pose_estimator():
    """Initialize pose estimation model."""
    cfg_options = dict(model=dict(test_cfg=dict(output_heatmaps=True)))
    pose_estimator = init_model(POSE_CONFIG, POSE_CHECKPOINT, device=DEVICE, cfg_options=cfg_options)
    return pose_estimator


def estimate_pose(pose_estimator, frame, bboxes):
    """Runs pose estimation and returns keypoint results."""
    pose_results = inference_topdown(pose_estimator, frame, bboxes)
    return merge_data_samples(pose_results), pose_results

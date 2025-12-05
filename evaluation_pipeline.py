from config.pipeline_config import PipelineConfig
from config.global_config import GlobalConfig
from dataset_files.HumanEva.humaneva_evaluation import run_humaneva_assessment
from dataset_files.MoVi.movi_evaluation import run_movi_assessment
from dataset_files.HumanSC3D.humansc3d_evaluation import run_humansc3d_assessment


def run_pose_assessment_pipeline(
    pipeline_config: PipelineConfig,
    global_config: GlobalConfig,
    output_dir: str,
    input_dir: str,
    min_bbox_confidence=None,
    min_keypoint_confidence=None,
):
    """
    Run pose assessment pipeline with configurable confidence filtering.

    Args:
        pipeline_config (PipelineConfig): Pipeline configuration
        global_config (GlobalConfig): Global configuration
        output_dir (str): Output directory for results
        input_dir (str): Input directory for data
        min_bbox_confidence (float or None): Minimum bounding box confidence threshold for filtering
        min_keypoint_confidence (float or None): Minimum keypoint confidence threshold for filtering
    """
    dataset = pipeline_config.paths.dataset or ""

    if dataset == "HumanEva":
        return run_humaneva_assessment(
            pipeline_config,
            global_config,
            output_dir,
            input_dir,
            min_bbox_confidence=min_bbox_confidence,
            min_keypoint_confidence=min_keypoint_confidence,
        )
    elif dataset == "MoVi":
        return run_movi_assessment(
            pipeline_config, global_config, output_dir, input_dir
        )
    elif dataset == "HumanSC3D":
        return run_humansc3d_assessment(
            pipeline_config,
            global_config,
            output_dir,
            input_dir,
            min_bbox_confidence=min_bbox_confidence,
            min_keypoint_confidence=min_keypoint_confidence,
        )
    else:
        raise NotImplementedError(f"Assessment not implemented for dataset '{dataset}'")

from config.pipeline_config import PipelineConfig
from config.global_config import GlobalConfig
from dataset_files.HumanEva.humaneva_evaluation import (
    run_humaneva_assessment,
)


def run_pose_assessment_pipeline(
    pipeline_config: PipelineConfig,
    global_config: GlobalConfig,
    output_dir: str,
    input_dir: str,
):
    dataset = pipeline_config.paths.dataset or ""

    if dataset == "HumanEva":
        return run_humaneva_assessment(
            pipeline_config, global_config, output_dir, input_dir
        )
    else:
        raise NotImplementedError(
            f"Assessment not implemented for dataset '{dataset}'")

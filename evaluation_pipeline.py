from config.pipeline_config import PipelineConfig
from config.global_config import GlobalConfig
from dataset_files.HumanEva.humaneva_evaluation import run_humaneva_assessment
from dataset_files.MoVi.movi_evaluation import run_movi_assessment  # ← Add this import


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
    elif dataset == "MoVi":
        return run_movi_assessment(
            pipeline_config, global_config, output_dir, input_dir
        )
    else:
        raise NotImplementedError(
            f"Assessment not implemented for dataset '{dataset}'"
        )

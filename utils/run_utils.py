import os
import shutil
from datetime import datetime
from pathlib import Path
import logging
import yaml  # Needed for saving config objects
from typing import Tuple

logger = logging.getLogger(__name__)


def get_pipeline_io_paths(global_paths, dataset: str) -> Tuple[Path, Path]:
    """
    Constructs full input and output paths for a specific dataset.

    Args:
        global_paths: the .paths section of the global config.
        dataset: name of the dataset to append to base input/output.

    Returns:
        Tuple[Path, Path]: (input_path, output_path)
    """
    input_path = Path(global_paths.input_dir) / dataset
    output_path = Path(global_paths.output_dir) / dataset

    os.makedirs(output_path, exist_ok=True)
    return input_path, output_path


def make_run_dir(
    base_out: str,
    pipeline_name: str,
    step_name: str,
    cfg_path: str,
    global_config_obj=None,
    pipeline_config_obj=None,
) -> Path:
    """
    Creates a timestamped directory for the pipeline run and optionally saves configs.

    Args:
        base_out: Base output directory.
        pipeline_name: Logical pipeline name (e.g., 'simulate_noise').
        step_name: Specific step name (e.g., 'noise', 'filter').
        cfg_path: Path to the pipeline config YAML (for backup copy).
        global_config_obj: Optional, to dump global config YAML.
        pipeline_config_obj: Optional, to dump pipeline config YAML.

    Returns:
        Path to the created run directory.
    """
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(base_out) / pipeline_name / step_name / ts
    original = run_dir
    count = 1
    while run_dir.exists():
        run_dir = Path(f"{original}_{count}")
        count += 1

    run_dir.mkdir(parents=True)

    # Optionally save structured config objects if provided
    if global_config_obj and hasattr(global_config_obj, "to_yaml"):
        global_config_obj.to_yaml(run_dir / "global_config.yaml")
    if pipeline_config_obj and hasattr(pipeline_config_obj, "to_yaml"):
        pipeline_config_obj.to_yaml(run_dir / "pipeline_config_structured.yaml")

    logger.info(f"Pipeline outputs will be saved under: {run_dir}")
    return run_dir

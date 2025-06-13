import shutil
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def make_run_dir(base_out: str, pipeline_name: str, cfg_path: str) -> Path:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(base_out) / pipeline_name / ts
    original = run_dir
    count = 1
    while run_dir.exists():
        run_dir = Path(f"{original}_{count}")
        count += 1

    run_dir.mkdir(parents=True)
    shutil.copy2(cfg_path, run_dir / "config.yaml")
    logger.info(f"Pipeline outputs will be saved under: {run_dir}")
    return run_dir

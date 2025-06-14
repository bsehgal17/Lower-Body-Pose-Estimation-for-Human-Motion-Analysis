import os
import yaml
from typing import Optional
from dataclasses import dataclass
import logging

# pipeline‑level paths (dataset + ground‑truth)
from .pipeline_path import PipelinePathsConfig
from .models_config import ModelsConfig
from .processing_config import ProcessingConfig
from .sync_data_config import SyncDataConfig
from .filter_config import FilterConfig
from .noise_config import NoiseConfig
from .evaluation_config import EvaluationConfig

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Full configuration for a specific processing pipeline."""

    paths: PipelinePathsConfig
    models: ModelsConfig
    processing: ProcessingConfig
    sync_data: SyncDataConfig
    filter: Optional[FilterConfig] = None
    noise: Optional[NoiseConfig] = None
    evaluation: Optional[EvaluationConfig] = None

    # ------------------------------------------------------------------
    # YAML (de)serialization helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "PipelineConfig":
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Pipeline config file not found: {yaml_path}")
        with open(yaml_path, "r", encoding="utf-8") as f:
            raw_config = yaml.safe_load(f)

        return cls(
            paths=PipelinePathsConfig(**raw_config.get("paths", {})),
            models=ModelsConfig(**raw_config.get("models", {})),
            processing=ProcessingConfig(**raw_config.get("processing", {})),
            sync_data=SyncDataConfig(data=raw_config.get("sync_data", {}).get("data")),
            filter=FilterConfig(**raw_config["filter"])
            if "filter" in raw_config
            else None,
            noise=NoiseConfig(**raw_config["noise"]) if "noise" in raw_config else None,
            evaluation=EvaluationConfig(**raw_config["evaluation"])
            if "evaluation" in raw_config
            else None,
        )

    def to_yaml(self, yaml_path: str):
        """Dump the current config to YAML."""
        cfg_dict = {
            "paths": self.paths.__dict__,
            "models": self.models.__dict__,
            "processing": self.processing.__dict__,
            "sync_data": {"data": self.sync_data.data},
        }
        if self.filter:
            cfg_dict["filter"] = self.filter.__dict__
        if self.noise:
            cfg_dict["noise"] = self.noise.__dict__
        if self.evaluation:
            cfg_dict["evaluation"] = self.evaluation.__dict__

        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(cfg_dict, f, indent=4)


# ------------------------------------------------------------
# Singleton accessor
# ------------------------------------------------------------
_pipeline_config: Optional[PipelineConfig] = None


def get_pipeline_config(config_file_path: Optional[str] = None) -> PipelineConfig:
    global _pipeline_config
    if _pipeline_config is None:
        if config_file_path:
            logger.info(f"Loading pipeline config from {config_file_path}")
            _pipeline_config = PipelineConfig.from_yaml(config_file_path)
        else:
            raise ValueError(
                "config_file_path must be provided for initial load of PipelineConfig."
            )
    return _pipeline_config

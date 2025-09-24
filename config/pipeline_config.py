import os
import yaml
from typing import Optional
from dataclasses import dataclass
import logging

# pipeline‑level paths (dataset + ground‑truth)
from .pipeline_path import PipelinePathsConfig
from .models_config import ModelsConfig
from .processing_config import ProcessingConfig
from .filter_config import FilterConfig
from .noise_config import NoiseConfig
from .evaluation_config import EvaluationConfig
from .dataset_config import DatasetConfig  # <- new import
from .enhancement_config import EnhancementConfig  # <- enhancement import
# <- confidence filtering import


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    paths: PipelinePathsConfig
    models: Optional[ModelsConfig] = None
    processing: Optional[ProcessingConfig] = None
    filter: Optional[FilterConfig] = None
    noise: Optional[NoiseConfig] = None
    evaluation: Optional[EvaluationConfig] = None
    dataset: Optional[DatasetConfig] = None
    enhancement: Optional[EnhancementConfig] = None

    # ------------------------------------------------------------------
    # YAML (de)serialization helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "PipelineConfig":
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(
                f"Pipeline config file not found: {yaml_path}")
        with open(yaml_path, "r", encoding="utf-8") as f:
            raw_config = yaml.safe_load(f)

        return cls(
            paths=PipelinePathsConfig(**raw_config.get("paths", {})),
            models=ModelsConfig(**raw_config["models"])
            if "models" in raw_config
            else None,
            processing=ProcessingConfig(**raw_config["processing"])
            if "processing" in raw_config
            else None,
            filter=FilterConfig(**raw_config["filter"])
            if "filter" in raw_config
            else None,
            noise=NoiseConfig(
                **raw_config["noise"]) if "noise" in raw_config else None,
            evaluation=EvaluationConfig(**raw_config["evaluation"])
            if "evaluation" in raw_config
            else None,
            dataset=DatasetConfig(**raw_config["dataset"])
            if "dataset" in raw_config
            else None,
            enhancement=cls._parse_enhancement_config(
                raw_config.get("enhancement"))
            if "enhancement" in raw_config
            else None,

        )

    @classmethod
    def _parse_enhancement_config(cls, enhancement_data):
        """Parse enhancement configuration data."""
        if not enhancement_data:
            return None

        from .enhancement_config import (
            EnhancementConfig,
            CLAHEConfig,
            FilteredCLAHEConfig,
            BrightnessConfig,
            BlurConfig,
            GammaConfig,
            FilteredGammaConfig,
            EnhancementProcessingConfig,
        )

        # Parse sub-configs
        clahe = None
        if "clahe" in enhancement_data:
            clahe = CLAHEConfig(**enhancement_data["clahe"])

        filtered_clahe = None
        if "filtered_clahe" in enhancement_data:
            filtered_clahe = FilteredCLAHEConfig(
                **enhancement_data["filtered_clahe"])

        brightness = None
        if "brightness" in enhancement_data:
            brightness = BrightnessConfig(**enhancement_data["brightness"])

        blur = None
        if "blur" in enhancement_data:
            blur = BlurConfig(**enhancement_data["blur"])

        gamma = None
        if "gamma" in enhancement_data:
            gamma = GammaConfig(**enhancement_data["gamma"])

        filtered_gamma = None
        if "filtered_gamma" in enhancement_data:
            filtered_gamma = FilteredGammaConfig(
                **enhancement_data["filtered_gamma"])

        processing = None
        if "processing" in enhancement_data:
            processing = EnhancementProcessingConfig(
                **enhancement_data["processing"])

        return EnhancementConfig(
            type=enhancement_data.get("type"),
            clahe=clahe,
            filtered_clahe=filtered_clahe,
            brightness=brightness,
            blur=blur,
            gamma=gamma,
            filtered_gamma=filtered_gamma,
            processing=processing,
            create_comparison_images=enhancement_data.get(
                "create_comparison_images"),
        )

    def to_yaml(self, yaml_path: str):
        """Dump the current config to YAML."""
        cfg_dict = {}

        if self.paths:
            cfg_dict["paths"] = self.paths.__dict__
        if self.models:
            cfg_dict["models"] = self.models.__dict__
        if self.processing:
            cfg_dict["processing"] = self.processing.__dict__
        if self.filter:
            cfg_dict["filter"] = self.filter.__dict__
        if self.noise:
            cfg_dict["noise"] = self.noise.__dict__
        if self.evaluation:
            cfg_dict["evaluation"] = self.evaluation.__dict__
        if self.dataset:
            cfg_dict["dataset"] = self.dataset.__dict__
        if self.enhancement:
            cfg_dict["enhancement"] = self.enhancement.__dict__

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

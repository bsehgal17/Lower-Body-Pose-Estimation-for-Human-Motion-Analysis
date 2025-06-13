# config.py
import os
import yaml
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field
import logging
from .paths_config import PathsConfig
from .video_config import VideoConfig
from .models_config import ModelsConfig
from .processing_config import ProcessingConfig
from .sync_data_config import SyncDataConfig
from .filter_config import FilterConfig
from .noise_config import NoiseConfig
from .assessment_config import AssessmentConfig


# Configure logging for the config module
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class GlobalConfig:
    """Centralized configuration for the entire application."""

    paths: PathsConfig = field(default_factory=PathsConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    sync_data: SyncDataConfig = field(default_factory=SyncDataConfig)
    filter: Optional[FilterConfig] = field(default_factory=FilterConfig)
    noise: Optional[NoiseConfig] = field(default_factory=NoiseConfig)
    assessment: Optional[AssessmentConfig] = field(
        default_factory=AssessmentConfig)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "GlobalConfig":
        """Loads configuration from a YAML file."""
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

        with open(yaml_path, "r") as f:
            raw_config = yaml.safe_load(f)

        # Deserialize into dataclasses
        paths_config = PathsConfig(**raw_config.get("paths", {}))
        video_config = VideoConfig(**raw_config.get("video", {}))
        models_config = ModelsConfig(**raw_config.get("models", {}))
        processing_config = ProcessingConfig(
            **raw_config.get("processing", {}))
        sync_data_config = SyncDataConfig(
            data=raw_config.get("sync_data", {}).get("data")
        )

        filter_config = FilterConfig(**raw_config.get("filter", {}))
        noise_config = NoiseConfig(**raw_config.get("noise", {}))
        assessment_config = AssessmentConfig(
            **raw_config.get("assessment", {}))

        return cls(
            paths=paths_config,
            video=video_config,
            models=models_config,
            processing=processing_config,
            sync_data=sync_data_config,
            filter=filter_config,
            noise=noise_config,
            assessment=assessment_config,
        )

    def to_yaml(self, yaml_path: str):
        config_dict = {
            "paths": self.paths.__dict__,
            "video": self.video.__dict__,
            "models": self.models.__dict__,
            "processing": self.processing.__dict__,
            "sync_data": {"data": self.sync_data.data},
            "noise": self.noise.__dict__,
            "filter": self.filter.__dict__,
            "assessment": self.assessment.__dict__,
        }
        with open(yaml_path, "w") as f:
            yaml.dump(config_dict, f, indent=4)


# --- 2. Centralized Configuration Management (Optional but Recommended) ---

# You can create a singleton instance or a function to get the config
# This prevents multiple initializations and ensures a single source of truth.
_global_config: Optional[GlobalConfig] = None


def get_config(config_file_path: Optional[str] = None) -> GlobalConfig:
    """
    Returns the singleton GlobalConfig instance.
    Loads from YAML if config_file_path is provided and config hasn't been loaded yet.
    """
    global _global_config
    if _global_config is None:
        if config_file_path:
            logger.info(f"Loading configuration from {config_file_path}")
            _global_config = GlobalConfig.from_yaml(config_file_path)
        else:
            logger.info(
                "No config file specified, using default configurations.")
            _global_config = GlobalConfig()
    return _global_config

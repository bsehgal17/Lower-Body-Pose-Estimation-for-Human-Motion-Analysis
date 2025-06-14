import os
import yaml
from typing import Optional
from dataclasses import dataclass, field
import logging

from .global_paths import GlobalPathsConfig
from .video_config import VideoConfig

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class GlobalConfig:
    """Holds settings shared across all pipelines — input/output paths, video types, etc."""

    paths: GlobalPathsConfig = field(default_factory=GlobalPathsConfig)
    video: VideoConfig = field(default_factory=VideoConfig)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "GlobalConfig":
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Global config not found: {yaml_path}")
        with open(yaml_path, "r") as f:
            raw_config = yaml.safe_load(f)

        return cls(
            paths=GlobalPathsConfig(**raw_config.get("paths", {})),
            video=VideoConfig(**raw_config.get("video", {})),
        )

    def to_yaml(self, yaml_path: str):
        config_dict = {
            "paths": self.paths.__dict__,
            "video": self.video.__dict__,
        }
        with open(yaml_path, "w") as f:
            yaml.dump(config_dict, f, indent=4)


_global_config: Optional[GlobalConfig] = None


def get_global_config(config_file_path: Optional[str] = None) -> GlobalConfig:
    global _global_config
    if _global_config is None:
        if config_file_path:
            logger.info(f"Loading global config from {config_file_path}")
            _global_config = GlobalConfig.from_yaml(config_file_path)
        else:
            logger.info("No global config path specified. Using defaults.")
            _global_config = GlobalConfig()
    return _global_config

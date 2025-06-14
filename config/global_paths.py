from dataclasses import dataclass
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class GlobalPathsConfig:
    input_dir: str
    output_dir: str

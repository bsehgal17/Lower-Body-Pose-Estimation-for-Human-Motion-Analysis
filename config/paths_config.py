from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field
import logging
import os

# Configure logging for the config module
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class PathsConfig:
    """Configuration for file paths."""

    video_folder: str = (
        "/storage/Projects/Gaitly/bsehgal/lower_body_pose_est/HumanEva_walking/"
    )
    output_dir: str = (
        "/storage/Projects/Gaitly/bsehgal/lower_body_pose_est/test_videos_results"
    )
    csv_file: str = "C:\\Users\\BhavyaSehgal\\Downloads\\bhavya_phd\\Tested_dataset\\humaneva_sorted_by_subject.csv"

    def __post_init__(self):
        # Ensure paths exist or are created (output_dir)
        if not os.path.exists(self.video_folder):
            logger.warning(f"Video folder does not exist: {self.video_folder}")
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"Output directory ensured: {self.output_dir}")
        if not os.path.exists(self.csv_file):
            logger.warning(f"CSV file does not exist: {self.csv_file}")

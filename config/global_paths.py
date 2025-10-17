from pydantic import BaseModel
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class GlobalPathsConfig(BaseModel):
    input_dir: str = None
    output_dir: str = None

from pydantic import BaseModel, Field
from typing import Optional, Dict


class DatasetConfig(BaseModel):
    """
    Configuration for dataset-specific settings and keypoint format definitions.

    Defines how pose estimation results should be interpreted and processed
    for different datasets with varying keypoint schemas and synchronization requirements.
    """

    keypoint_format: Optional[str] = Field(
        default=None,
        description="String identifier for the keypoint format/schema used in this dataset. "
        "Examples: 'coco_17', 'coco_133', 'h36m_17', 'mpii_16'. "
        "Used to determine joint ordering and interpretation of pose data.",
    )

    joint_enum_module: Optional[str] = Field(
        default=None,
        description="Python module path containing joint enumeration constants. "
        "Example: 'utils.joint_enum' which would contain joint name mappings "
        "like NOSE=0, LEFT_EYE=1, etc. for programmatic joint access.",
    )

    sync_data: Optional[Dict[str, Dict[str, Dict[str, list]]]] = Field(
        default=None,
        description="Synchronization data for multi-view or multi-sequence datasets. "
        "Structure: {subject_id: {action_name: [frame_indices]}} "
        "Example: {'S1': {'Walking': [10, 12, 14]}} indicates frames "
        "10, 12, 14 are synchronized across cameras for subject S1 walking. "
        "Used for temporal alignment in multi-camera setups.",
    )

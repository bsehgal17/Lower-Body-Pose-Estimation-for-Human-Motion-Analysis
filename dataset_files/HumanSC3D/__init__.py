"""
HumanSC3D Dataset Handler Module

This module provides functionality for handling the HumanSC3D dataset
in the Lower-Body Pose Estimation pipeline.

Classes:
    HumanSC3DGroundTruthLoader: Loads ground truth keypoints from JSON files

Functions:
    get_humansc3d_metadata_from_video: Extract metadata from video paths
    get_humansc3d_gt_path: Generate ground truth file paths
    load_humansc3d_gt_keypoints: Convenience function for loading keypoints
    humansc3d_data_loader: Data loader for evaluation pipeline
    run_humansc3d_assessment: Main evaluation function
"""

from .humansc3d_metadata import get_humansc3d_metadata_from_video, get_humansc3d_gt_path

from .humansc3d_gt_loader import HumanSC3DGroundTruthLoader, load_humansc3d_gt_keypoints

from .humansc3d_evaluation import humansc3d_data_loader, run_humansc3d_assessment

__all__ = [
    "get_humansc3d_metadata_from_video",
    "get_humansc3d_gt_path",
    "HumanSC3DGroundTruthLoader",
    "load_humansc3d_gt_keypoints",
    "humansc3d_data_loader",
    "run_humansc3d_assessment",
]

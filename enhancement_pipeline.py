"""
Enhancement Pipeline for Video Processing

This module provides a pipeline for applying image/frame enhancement techniques
(like CLAHE) to video datasets, following the same pattern as detection,
filtering, and evaluation pipelines.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Union
import cv2
import numpy as np
from tqdm import tqdm

from config.pipeline_config import PipelineConfig
from config.global_config import GlobalConfig
from utils.video_io import get_video_files

logger = logging.getLogger(__name__)


class EnhancementPipeline:
    """
    Enhancement pipeline for applying image enhancement techniques to videos.
    """

    def __init__(self, pipeline_config: PipelineConfig, global_config: GlobalConfig):
        """Initialize the enhancement pipeline."""
        self.pipeline_config = pipeline_config
        self.global_config = global_config
        self.enhancement_config = self._extract_enhancement_config()

        logger.info("Enhancement pipeline initialized")

    def _extract_enhancement_config(self):
        """Extract enhancement configuration from pipeline config."""
        if hasattr(self.pipeline_config, "enhancement"):
            return self.pipeline_config.enhancement
        else:
            # No default config - must be explicitly specified
            raise ValueError(
                "Pipeline configuration must contain 'enhancement' section"
            )

    def process_videos(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        enhancement_type: str,
    ) -> bool:
        """
        Process all videos in input directory with specified enhancement.

        Args:
            input_dir: Directory containing input videos
            output_dir: Directory to save enhanced videos
            enhancement_type: Type of enhancement ('clahe', 'histogram_eq', etc.)

        Returns:
            bool: True if processing successful, False otherwise
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        if not input_dir.exists():
            logger.error(f"Input directory not found: {input_dir}")
            return False

        output_dir.mkdir(parents=True, exist_ok=True)

        # Get video files with common video extensions
        video_extensions = [".mp4", ".avi", ".mov",
                            ".mkv", ".wmv", ".flv", ".webm"]
        video_files = get_video_files(str(input_dir), video_extensions)
        if not video_files:
            logger.warning(f"No video files found in {input_dir}")
            return False  # Changed from True to False - no videos should be considered failure

        logger.info(f"Found {len(video_files)} videos to process")
        logger.info(f"Enhancement type: {enhancement_type}")
        logger.info(f"Input: {input_dir}")
        logger.info(f"Output: {output_dir}")
        logger.info("Folder structure will be preserved in output directory")

        success_count = 0

        for video_file in tqdm(video_files, desc="Enhancing videos"):
            video_path = Path(video_file)

            # Calculate relative path from input directory to maintain folder structure
            try:
                relative_path = video_path.relative_to(input_dir)
                # Create the same folder structure in output directory
                output_path = (
                    output_dir / relative_path.parent / f"{relative_path.name}"
                )
                output_path.parent.mkdir(parents=True, exist_ok=True)
                logger.debug(
                    f"Processing {relative_path} -> {output_path.relative_to(output_dir)}"
                )
            except ValueError:
                # If relative_to fails, fall back to simple naming
                output_path = output_dir / f"{video_path.name}"
                logger.debug(
                    f"Processing {video_path.name} -> {output_path.name}")

            success = self._apply_enhancement(
                video_path, output_path, enhancement_type)

            if success:
                success_count += 1
                logger.debug(f"Successfully enhanced: {video_path.name}")
            else:
                logger.error(f"Failed to enhance: {video_path.name}")

        logger.info(
            f"Enhancement completed: {success_count}/{len(video_files)} videos processed successfully"
        )
        return success_count == len(video_files)

    def _apply_enhancement(
        self, input_path: Path, output_path: Path, enhancement_type: str
    ) -> bool:
        """Apply the specified enhancement to a single video."""
        try:
            # Get enhancement function from registry
            from filtering_and_data_cleaning.enhancement_registry import (
                get_enhancement_function,
            )

            enhancement_func = get_enhancement_function(enhancement_type)

            # Prepare enhancement parameters based on type and config
            kwargs = self._get_enhancement_parameters(enhancement_type)

            success = enhancement_func(
                video_path=input_path, output_path=output_path, **kwargs
            )
            return success
        except Exception as e:
            logger.error(
                f"{enhancement_type} enhancement failed for {input_path}: {e}")
            return False

    def _get_enhancement_parameters(self, enhancement_type: str) -> dict:
        """Get parameters for the specified enhancement type from config."""
        params = {}

        if enhancement_type == "clahe":
            if hasattr(self.enhancement_config, "clahe"):
                params["clip_limit"] = self.enhancement_config.clahe.clip_limit
                params["tile_grid_size"] = self.enhancement_config.clahe.tile_grid_size
                params["color_space"] = getattr(
                    self.enhancement_config.clahe, "color_space", "LAB"
                )
            else:
                raise ValueError(
                    "CLAHE enhancement requires 'clahe' section in configuration"
                )

        elif enhancement_type == "brightness_adjustment":
            if hasattr(self.enhancement_config, "brightness"):
                params["factor"] = self.enhancement_config.brightness.factor
            else:
                raise ValueError(
                    "Brightness adjustment requires 'brightness' section in configuration"
                )

        elif enhancement_type == "gaussian_blur":
            if hasattr(self.enhancement_config, "blur"):
                params["kernel_size"] = tuple(
                    self.enhancement_config.blur.kernel_size)
                params["sigma_x"] = getattr(
                    self.enhancement_config.blur, "sigma_x", 0)
                params["sigma_y"] = getattr(
                    self.enhancement_config.blur, "sigma_y", 0)
            else:
                raise ValueError(
                    "Gaussian blur requires 'blur' section in configuration"
                )

        # histogram_eq doesn't need parameters

        return params

    def process_dataset_structure(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        enhancement_type: str,
    ) -> bool:
        """
        Process videos organized in dataset structure (subject/action/videos).

        Args:
            input_dir: Root directory of dataset
            output_dir: Root directory for enhanced dataset
            enhancement_type: Type of enhancement to apply

        Returns:
            bool: True if processing successful
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        if not input_dir.exists():
            logger.error(f"Input directory not found: {input_dir}")
            return False

        output_dir.mkdir(parents=True, exist_ok=True)

        total_videos = 0
        processed_videos = 0

        # Walk through dataset structure
        for subject_dir in input_dir.iterdir():
            if not subject_dir.is_dir():
                continue

            subject_output_dir = output_dir / subject_dir.name
            subject_output_dir.mkdir(parents=True, exist_ok=True)

            for action_dir in subject_dir.iterdir():
                if not action_dir.is_dir():
                    continue

                action_output_dir = subject_output_dir / action_dir.name
                action_output_dir.mkdir(parents=True, exist_ok=True)

                # Process videos in this action directory
                video_extensions = [
                    ".mp4",
                    ".avi",
                    ".mov",
                    ".mkv",
                    ".wmv",
                    ".flv",
                    ".webm",
                ]
                video_files = get_video_files(
                    str(action_dir), video_extensions)
                total_videos += len(video_files)

                for video_file in video_files:
                    video_path = Path(video_file)
                    output_path = action_output_dir / f"{video_path.name}"

                    logger.info(
                        f"Processing: {subject_dir.name}/{action_dir.name}/{video_path.name}"
                    )

                    success = self._apply_enhancement(
                        video_path, output_path, enhancement_type
                    )

                    if success:
                        processed_videos += 1

        logger.info(
            f"Dataset enhancement completed: {processed_videos}/{total_videos} videos processed"
        )
        return processed_videos == total_videos

    def generate_enhancement_report(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        report_path: Union[str, Path],
    ) -> bool:
        """
        Generate a report comparing original and enhanced videos.

        Args:
            input_dir: Directory with original videos
            output_dir: Directory with enhanced videos
            report_path: Path to save the report

        Returns:
            bool: True if report generated successfully
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        report_path = Path(report_path)

        video_extensions = [".mp4", ".avi", ".mov",
                            ".mkv", ".wmv", ".flv", ".webm"]
        original_videos = get_video_files(str(input_dir), video_extensions)
        enhanced_videos = get_video_files(str(output_dir), video_extensions)

        report_data = {
            "original_count": len(original_videos),
            "enhanced_count": len(enhanced_videos),
            "video_comparisons": [],
        }

        # Compare statistics for each video pair
        for orig_video in original_videos:
            orig_path = Path(orig_video)

            # Calculate the enhanced video path maintaining folder structure
            try:
                relative_path = orig_path.relative_to(input_dir)
                enhanced_path = (
                    output_dir / relative_path.parent / f"{relative_path.name}"
                )
            except ValueError:
                # If relative_to fails, fall back to simple naming
                enhanced_path = output_dir / f"{orig_path.name}"

            if enhanced_path.exists():
                stats = self._compare_video_statistics(
                    orig_path, enhanced_path)
                stats["video_name"] = (
                    str(relative_path)
                    if "relative_path" in locals()
                    else orig_path.name
                )
                report_data["video_comparisons"].append(stats)

        # Save report
        import json

        with open(report_path, "w") as f:
            json.dump(report_data, f, indent=2)

        logger.info(f"Enhancement report saved to: {report_path}")
        return True

    def _compare_video_statistics(
        self, original_path: Path, enhanced_path: Path
    ) -> Dict[str, Any]:
        """Compare statistics between original and enhanced video."""
        stats = {
            "original_file": str(original_path),
            "enhanced_file": str(enhanced_path),
            "frame_count_original": 0,
            "frame_count_enhanced": 0,
            "avg_brightness_change": 0.0,
            "avg_contrast_change": 0.0,
        }

        try:
            # Analyze first 50 frames for performance
            cap_orig = cv2.VideoCapture(str(original_path))
            cap_enh = cv2.VideoCapture(str(enhanced_path))

            if not cap_orig.isOpened() or not cap_enh.isOpened():
                return stats

            stats["frame_count_original"] = int(
                cap_orig.get(cv2.CAP_PROP_FRAME_COUNT))
            stats["frame_count_enhanced"] = int(
                cap_enh.get(cv2.CAP_PROP_FRAME_COUNT))

            brightness_changes = []
            contrast_changes = []
            frame_count = 0

            while frame_count < 50:  # Analyze first 50 frames
                ret_orig, frame_orig = cap_orig.read()
                ret_enh, frame_enh = cap_enh.read()

                if not ret_orig or not ret_enh:
                    break

                # Convert to grayscale for analysis
                gray_orig = cv2.cvtColor(frame_orig, cv2.COLOR_BGR2GRAY)
                gray_enh = cv2.cvtColor(frame_enh, cv2.COLOR_BGR2GRAY)

                # Calculate brightness (mean)
                brightness_orig = np.mean(gray_orig)
                brightness_enh = np.mean(gray_enh)
                brightness_changes.append(brightness_enh - brightness_orig)

                # Calculate contrast (standard deviation)
                contrast_orig = np.std(gray_orig)
                contrast_enh = np.std(gray_enh)
                contrast_changes.append(contrast_enh - contrast_orig)

                frame_count += 1

            cap_orig.release()
            cap_enh.release()

            if brightness_changes:
                stats["avg_brightness_change"] = float(
                    np.mean(brightness_changes))
                stats["avg_contrast_change"] = float(np.mean(contrast_changes))

        except Exception as e:
            logger.error(f"Error comparing videos: {e}")

        return stats


def run_enhancement_pipeline(
    pipeline_config: PipelineConfig,
    global_config: GlobalConfig,
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    enhancement_type: str,
    dataset_structure: bool,
) -> bool:
    """
    Main function to run the enhancement pipeline.

    Args:
        pipeline_config: Pipeline configuration
        global_config: Global configuration
        input_dir: Input directory containing videos
        output_dir: Output directory for enhanced videos
        enhancement_type: Type of enhancement ('clahe', 'histogram_eq')
        dataset_structure: Whether to process as structured dataset

    Returns:
        bool: True if successful, False otherwise
    """
    logger.info("Starting enhancement pipeline")

    pipeline = EnhancementPipeline(pipeline_config, global_config)

    if dataset_structure:
        success = pipeline.process_dataset_structure(
            input_dir, output_dir, enhancement_type
        )
    else:
        success = pipeline.process_videos(
            input_dir, output_dir, enhancement_type)

    # Generate report
    report_path = Path(output_dir) / "enhancement_report.json"
    pipeline.generate_enhancement_report(input_dir, output_dir, report_path)

    if success:
        logger.info("Enhancement pipeline completed successfully")
    else:
        logger.error("Enhancement pipeline completed with errors")

    return success

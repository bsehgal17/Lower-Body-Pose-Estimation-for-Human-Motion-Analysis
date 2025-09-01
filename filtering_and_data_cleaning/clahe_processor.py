"""
CLAHE Pipeline Processor

This module provides a pipeline processor for applying CLAHE enhancement to video datasets,
integrating with the existing project structure and configuration system.
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
import cv2
import numpy as np

from ..config.enhancement_config import CLAHEConfig
from .clahe_enhancement import (
    CLAHEEnhancer,
    batch_enhance_videos,
)

logger = logging.getLogger(__name__)


class CLAHEProcessor:
    """
    A processor class for applying CLAHE enhancement to video datasets based on YAML configuration.
    """

    def __init__(self, config_path: Union[str, Path]):
        """
        Initialize the CLAHE processor with a configuration file.

        Args:
            config_path (Union[str, Path]): Path to the YAML configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.clahe_config = self._extract_clahe_config()
        self.enhancer = CLAHEEnhancer(
            clip_limit=self.clahe_config.clip_limit,
            tile_grid_size=self.clahe_config.tile_grid_size,
        )

        logger.info(f"CLAHE processor initialized with config: {config_path}")

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, "r") as file:
            config = yaml.safe_load(file)

        logger.info(f"Configuration loaded from: {self.config_path}")
        return config

    def _extract_clahe_config(self) -> CLAHEConfig:
        """
        Extracts and validates the CLAHE configuration from the loaded config.
        This method enforces that all necessary parameters are explicitly defined
        in the YAML file, as it does not use hardcoded default values.
        """
        if "clahe" not in self.config:
            raise ValueError(
                "Configuration error: 'clahe' section is missing from the YAML file."
            )

        clahe_section = self.config["clahe"]

        # Use global paths as a fallback for input/output directories
        if "input_dir" not in clahe_section:
            clahe_section["input_dir"] = self.config.get("paths", {}).get("input_dir")
        if "output_dir" not in clahe_section:
            clahe_section["output_dir"] = self.config.get("paths", {}).get("output_dir")

        try:
            # Directly create the config object; this will fail if keys are missing
            clahe_config = CLAHEConfig(**clahe_section)

            # Ensure tile_grid_size and file_extensions are tuples
            clahe_config.tile_grid_size = tuple(clahe_config.tile_grid_size)
            clahe_config.file_extensions = tuple(clahe_config.file_extensions)

        except TypeError as e:
            # Catches errors from missing arguments in the dataclass constructor
            raise ValueError(
                f"Missing required parameter in 'clahe' section of the config file. Details: {e}"
            )

        logger.info(f"CLAHE config extracted: {clahe_config}")
        return clahe_config

    def process_single_video(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
    ) -> bool:
        """
        Process a single video file with CLAHE enhancement.

        Args:
            input_path (Union[str, Path]): Path to input video
            output_path (Optional[Union[str, Path]]): Path to output video.
                                                     If None, auto-generate based on input.

        Returns:
            bool: True if successful, False otherwise
        """
        input_path = Path(input_path)

        if not input_path.exists():
            logger.error(f"Input video not found: {input_path}")
            return False

        # Auto-generate output path if not provided
        if output_path is None:
            if self.clahe_config.output_dir:
                output_dir = Path(self.clahe_config.output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / f"clahe_{input_path.name}"
            else:
                output_path = input_path.parent / f"clahe_{input_path.name}"
        else:
            output_path = Path(output_path)

        logger.info(f"Processing video: {input_path} -> {output_path}")

        def progress_callback(frame_num, total_frames, progress):
            if frame_num % 30 == 0:  # Log every 30 frames
                logger.info(
                    f"Progress: {progress:.1f}% ({frame_num}/{total_frames} frames)"
                )

        success = self.enhancer.enhance_video(
            input_path=input_path,
            output_path=output_path,
            color_space=self.clahe_config.color_space,
            progress_callback=progress_callback,
        )

        if success:
            logger.info(f"Successfully enhanced video: {output_path}")
        else:
            logger.error(f"Failed to enhance video: {input_path}")

        return success

    def process_batch(self) -> bool:
        """
        Process all videos in the configured input directory.

        Returns:
            bool: True if all videos processed successfully, False otherwise
        """
        if not self.clahe_config.input_dir:
            logger.error("Input directory not specified in configuration")
            return False

        if not self.clahe_config.output_dir:
            logger.error("Output directory not specified in configuration")
            return False

        input_dir = Path(self.clahe_config.input_dir)
        output_dir = Path(self.clahe_config.output_dir)

        if not input_dir.exists():
            logger.error(f"Input directory not found: {input_dir}")
            return False

        logger.info(f"Starting batch processing: {input_dir} -> {output_dir}")

        try:
            batch_enhance_videos(
                input_dir=input_dir,
                output_dir=output_dir,
                clip_limit=self.clahe_config.clip_limit,
                tile_grid_size=self.clahe_config.tile_grid_size,
                color_space=self.clahe_config.color_space,
                file_extensions=self.clahe_config.file_extensions,
            )

            logger.info("Batch processing completed successfully")
            return True

        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            return False

    def process_dataset(self, dataset_config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Process videos based on dataset configuration structure.

        Args:
            dataset_config (Optional[Dict[str, Any]]): Dataset configuration.
                                                      If None, uses config from file.

        Returns:
            bool: True if successful, False otherwise
        """
        if dataset_config is None:
            dataset_config = self.config.get("dataset", {})

        if not dataset_config:
            logger.warning(
                "No dataset configuration found, falling back to batch processing"
            )
            return self.process_batch()

        # Handle dataset-specific structure (e.g., HumanEva, MoVi)
        dataset_name = self.config.get("paths", {}).get("dataset", "unknown")
        logger.info(f"Processing dataset: {dataset_name}")

        if "sync_data" in dataset_config:
            return self._process_structured_dataset(dataset_config["sync_data"])
        else:
            return self.process_batch()

    def _process_structured_dataset(self, sync_data: Dict[str, Any]) -> bool:
        """
        Process a structured dataset with subject/action organization.

        Args:
            sync_data (Dict[str, Any]): Synchronized data configuration

        Returns:
            bool: True if successful, False otherwise
        """
        base_input_dir = (
            Path(self.clahe_config.input_dir) if self.clahe_config.input_dir else None
        )
        base_output_dir = (
            Path(self.clahe_config.output_dir) if self.clahe_config.output_dir else None
        )

        if not base_input_dir or not base_output_dir:
            logger.error(
                "Input and output directories must be specified for structured dataset"
            )
            return False

        success_count = 0
        total_count = 0

        # Process each subject
        for subject, actions in sync_data.get("data", {}).items():
            subject_input_dir = base_input_dir / subject
            subject_output_dir = base_output_dir / subject

            if not subject_input_dir.exists():
                logger.warning(f"Subject directory not found: {subject_input_dir}")
                continue

            subject_output_dir.mkdir(parents=True, exist_ok=True)

            # Process each action for the subject
            for action, sync_values in actions.items():
                action_input_dir = subject_input_dir / action
                action_output_dir = subject_output_dir / action

                if not action_input_dir.exists():
                    logger.warning(f"Action directory not found: {action_input_dir}")
                    continue

                action_output_dir.mkdir(parents=True, exist_ok=True)

                # Find video files in action directory
                video_files = []
                for ext in self.clahe_config.file_extensions:
                    video_files.extend(action_input_dir.glob(f"*{ext}"))

                # Process each video file
                for video_file in video_files:
                    total_count += 1
                    output_file = action_output_dir / f"clahe_{video_file.name}"

                    logger.info(f"Processing: {subject}/{action}/{video_file.name}")

                    if self.process_single_video(video_file, output_file):
                        success_count += 1
                    else:
                        logger.error(f"Failed to process: {video_file}")

        logger.info(
            f"Structured dataset processing completed: {success_count}/{total_count} videos processed successfully"
        )
        return success_count == total_count

    def get_enhancement_statistics(
        self, input_path: Union[str, Path], output_path: Union[str, Path]
    ) -> Dict[str, Any]:
        """
        Calculate enhancement statistics by comparing input and output videos.

        Args:
            input_path (Union[str, Path]): Path to original video
            output_path (Union[str, Path]): Path to enhanced video

        Returns:
            Dict[str, Any]: Statistics including contrast improvement, brightness changes, etc.
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        if not input_path.exists() or not output_path.exists():
            logger.error(
                "Both input and output videos must exist for statistics calculation"
            )
            return {}

        # Open both videos
        cap_input = cv2.VideoCapture(str(input_path))
        cap_output = cv2.VideoCapture(str(output_path))

        if not cap_input.isOpened() or not cap_output.isOpened():
            logger.error("Could not open one or both videos for statistics")
            return {}

        stats = {
            "frames_processed": 0,
            "contrast_improvements": [],
            "brightness_changes": [],
            "histogram_differences": [],
        }

        try:
            while True:
                ret_input, frame_input = cap_input.read()
                ret_output, frame_output = cap_output.read()

                if not ret_input or not ret_output:
                    break

                # Convert to LAB for analysis
                lab_input = cv2.cvtColor(frame_input, cv2.COLOR_BGR2LAB)
                lab_output = cv2.cvtColor(frame_output, cv2.COLOR_BGR2LAB)

                # Calculate contrast (std of L channel)
                contrast_input = np.std(lab_input[:, :, 0])
                contrast_output = np.std(lab_output[:, :, 0])
                contrast_improvement = contrast_output - contrast_input

                # Calculate brightness (mean of L channel)
                brightness_input = np.mean(lab_input[:, :, 0])
                brightness_output = np.mean(lab_output[:, :, 0])
                brightness_change = brightness_output - brightness_input

                # Calculate histogram difference
                hist_input = cv2.calcHist(
                    [lab_input[:, :, 0]], [0], None, [256], [0, 256]
                )
                hist_output = cv2.calcHist(
                    [lab_output[:, :, 0]], [0], None, [256], [0, 256]
                )
                hist_diff = cv2.compareHist(hist_input, hist_output, cv2.HISTCMP_CHISQR)

                stats["contrast_improvements"].append(contrast_improvement)
                stats["brightness_changes"].append(brightness_change)
                stats["histogram_differences"].append(hist_diff)
                stats["frames_processed"] += 1

                # Limit analysis to first 100 frames for performance
                if stats["frames_processed"] >= 100:
                    break

        finally:
            cap_input.release()
            cap_output.release()

        # Calculate summary statistics
        if stats["frames_processed"] > 0:
            stats["avg_contrast_improvement"] = np.mean(stats["contrast_improvements"])
            stats["avg_brightness_change"] = np.mean(stats["brightness_changes"])
            stats["avg_histogram_difference"] = np.mean(stats["histogram_differences"])
            stats["contrast_improvement_std"] = np.std(stats["contrast_improvements"])
            stats["brightness_change_std"] = np.std(stats["brightness_changes"])

        return stats

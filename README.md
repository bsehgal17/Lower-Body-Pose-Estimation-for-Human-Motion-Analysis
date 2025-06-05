Configuration System (config/)
The configuration module defines and manages all settings used across the application in a structured and centralized way. Configuration is organized using Python dataclasses and loaded from a YAML file.

Structure
GlobalConfig acts as the root configuration container.

It includes sub-configs for:

PathsConfig: input/output folder paths and dataset locations.

VideoConfig: video file handling (extensions).

ModelsConfig: paths to model configs and checkpoints.

ProcessingConfig: common runtime settings (device, thresholds).

SyncDataConfig: subject-wise frame sync info for the dataset.

FilterConfig, NoiseConfig, AssessmentConfig: optional pipeline module configs.

Usage
A singleton pattern is used to ensure only one config instance is loaded:

python
Copy
Edit
from config.base import get_config
config = get_config("config.yaml")
The YAML file defines the config structure. Example:

yaml
Copy
Edit
paths:
  video_folder: ./data/videos
  output_dir: ./outputs
models:
  det_config: path/to/detector.py
  det_checkpoint: path/to/detector.pth
  pose_config: path/to/pose.py
  pose_checkpoint: path/to/pose.pth
This modular approach makes it easy to extend configurations when new pipeline stages (e.g. filtering, assessment) are added.

Pose Estimation Pipeline (pose_estimation/)
This module handles video-based human pose estimation using a two-stage approach: detection followed by top-down pose estimation.

Key Components
pipeline.py: Orchestrates the full detection pipeline for videos. It loads videos, initializes models, processes each frame, visualizes keypoints, and saves outputs.

detector.py: Initializes and runs an MMDetection model to detect bounding boxes for humans in each frame.

estimator.py: Wraps the MMPose inference API to estimate 2D keypoints from bounding boxes.

visualizer.py: Draws keypoints and skeletons on video frames using MMPose’s visualization tools.

processors/:

frame_processor.py: Handles the full processing of each frame (detection → pose estimation → visualization).

video_loader.py: Manages reading/writing video files using OpenCV.

Output
Annotated video files (with pose overlays)

JSON files containing pose keypoints for each frame

Example Entry Point
bash
Copy
Edit
python run.py detect --video_folder ./data/videos --output_dir ./results
Noise Simulation (processing/noise_simulator.py)
This module adds synthetic noise to video frames to simulate degraded input data, which can help evaluate robustness of pose estimation.

Features
Adds realistic Poisson (shot noise) and Gaussian (read noise).

Applies motion blur to simulate movement artifacts.

Optionally reduces brightness to simulate low-light conditions.

Supports per-frame transformation and writes a new noisy video.

Usage
Called via CLI:

bash
Copy
Edit
python run.py noise --input_folder ./data/videos --output_folder ./data/noisy_videos
Internally uses NoiseSimulator which reads configuration values like:

yaml
Copy
Edit
noise:
  poisson_scale: 1.5
  gaussian_std: 8
  motion_blur_kernel_size: 7
  apply_brightness_reduction: true
  brightness_factor: 20
This module is self-contained and can be enabled/disabled independently of other steps.

Utilities (utils/)
The utils/ directory contains helper functions and wrappers for common operations used throughout the pipeline.

Submodules
video_io.py

get_video_files(): Recursively finds all video files in a given folder with supported extensions.

frame_generator(): Yields frames from a video one-by-one.

json_io.py

combine_keypoints(): Aggregates frame-wise keypoints for saving.

save_keypoints_to_json(): Writes pose keypoints to a structured JSON file.

plotting.py

plot_filtering_effect(): Visual comparison of raw vs. filtered signals (used for debugging filtering).

Example Usage
python
Copy
Edit
from utils.video_io import get_video_files
from utils.json_io import save_keypoints_to_json
These helpers are designed to be stateless, reusable, and testable. They will support the upcoming modules for filtering and assessment as well.

Upcoming Modules (To Be Integrated)
The following pipeline components will be added soon:

Filtering
Apply smoothing to pose keypoints using techniques like:

Assessment
Evaluate the quality of pose estimations using metrics like

Each of these will have their own configuration sections, CLI commands, and modular handlers.
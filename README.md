---

# Lower-Body Pose Estimation Pipeline

This project provides a modular and extensible pipeline for performing pose estimation on video data, with support for noise simulation, visualization, and configuration-driven execution. The pipeline is designed for research-grade applications with flexibility in components, configurable settings, and CLI-based orchestration.

---

## Table of Contents

* [Overview](#overview)
* [Project Structure](#project-structure)
* [Installation](#installation)
* [Configuration Guidelines](#configuration-guidelines)

  * [1. Pose Estimation](#1-pose-estimation)
  * [2. Noise Simulation](#2-noise-simulation)
  * [3. Input/Output Paths](#3-inputoutput-paths)
  * [4. Video Handling](#4-video-handling)
* [Modules](#modules)

  * [Configuration](#configuration)
  * [Pose Estimation](#pose-estimation)
  * [Utilities](#utilities)
  * [Noise Simulation](#noise-simulation)

* [Main Script](#main-script)
* [Usage](#usage)

  * [CLI Commands](#cli-commands)

    * [`detect`](#detect)
    * [`noise`](#noise)
    * [`filter` *(coming soon)*](#filter-coming-soon)
    * [`assess` *(coming soon)*](#assess-coming-soon)

---

## Overview

The pipeline takes raw video input, performs human pose detection and keypoint estimation using MMPose and MMDetection models, optionally applies simulated camera noise, and saves both visualized videos and keypoint data. All behaviors are controlled by a YAML-based configuration file.

---

## Project Structure

```
project_root/
├── config/                     # Configuration modules and dataclasses
├── pose_estimation/           # Pose estimation models and logic
│   ├── detector.py
│   ├── estimator.py
│   ├── visualizer.py
│   ├── processors/
│   │   ├── frame_processor.py
│   │   └── video_loader.py
├── processing/
│   └── noise_simulator.py     # Video noise simulation
├── utils/
│   ├── video_io.py
│   ├── json_io.py
│   └── plotting.py
├── main.py                    # Entrypoint script
├── main_handlers.py           # CLI command handlers
├── cli.py                     # CLI command parser
├── run.py                     # Alternate entry point for CLI
└── config.yaml                # Example configuration
```

---

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/lower-body-pose-estimation.git
   cd lower-body-pose-estimation
   ```

2. Install required packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Make sure you have the necessary MMPose and MMDetection dependencies installed.

---

## Configuration Guidelines

All pipeline components are controlled through a centralized YAML file. Below is a detailed guide on what users should configure for each part of the pipeline and where these settings belong in the YAML file. 

All major settings (paths, models, thresholds, noise parameters) are defined in the YAML file. CLI arguments (if provided) can optionally override some of these values at runtime. If you prefer to rely entirely on YAML for input configuration, follow the structure below.

 However, the following CLI arguments are available to override specific fields defined in the YAML:

 #### Supported CLI Overrides

 * `--config_file <path>`

   * Path to the YAML config (default is usually hardcoded in `get_config()`).
 * For `detect` command:

   * `--video_folder <path>`
     Overrides `paths.video_folder` from YAML.
   * `--output_dir <path>`
     Overrides `paths.output_dir` from YAML.
 * For `noise` command:

   * `--input_folder <path>`
     Overrides `paths.video_folder`.
   * `--output_folder <path>`
     Overrides `paths.output_dir`.

 *(Note: Other internal config fields like models, thresholds, noise types, etc. must be edited directly in the YAML.)*

---

### 1. Pose Estimation

**Purpose**: Configure the models for human detection and pose estimation.

**YAML section**: `models` and `processing`

#### Required Edits in `models`:

```yaml
models:
  det_config: "path/to/mmdection_config.py"
  det_checkpoint: "path/to/detector_checkpoint.pth"
  pose_config: "path/to/mmpose_config.py"
  pose_checkpoint: "path/to/pose_checkpoint.pth"
```

#### Required Edits in `processing`:

```yaml
processing:
  device: "cuda:0"
  detection_threshold: 0.3
  nms_threshold: 0.3
  kpt_threshold: 0.3
```

**What to change**:

* Ensure all config and checkpoint paths are correct.
* Use `"cpu"` if GPU is not available.

---

### 2. Noise Simulation

**Purpose**: Add artificial camera noise, motion blur, and simulate poor lighting.

**YAML section**: `noise`

```yaml
noise:
  poisson_scale: 1.0
  gaussian_std: 5.0
  motion_blur_kernel_size: 5
  apply_brightness_reduction: true
  brightness_factor: 30
  target_resolution: [1280, 720]
```

**What to change**:

* Adjust `poisson_scale`, `gaussian_std`, and `brightness_factor` for desired effects.
* Set `target_resolution` only if resizing is needed.

---

### 3. Input/Output Paths

**Purpose**: Set source of videos and where to save outputs.

**YAML section**: `paths`

```yaml
paths:
  video_folder: "./videos/input"
  output_dir: "./videos/output"
  csv_file: "./metadata.csv"
```

**What to change**:

* Point `video_folder` to your input videos.
* Change `output_dir` to a desired save location.
* `csv_file` is optional (e.g., for HumanEva sync).

---

### 4. Video Handling

**Purpose**: Define which video formats to process.

**YAML section**: `video`

```yaml
video:
  extensions: [".mp4", ".avi", ".mov"]
```

**What to change**:

* Add/remove extensions based on your input file types.

---

### Full Example YAML

```yaml
paths:
  video_folder: "./videos"
  output_dir: "./results"
  csv_file: "./humaneva.csv"

video:
  extensions: [".mp4", ".avi"]

models:
  det_config: "checkpoints/mmdet_config.py"
  det_checkpoint: "checkpoints/detector.pth"
  pose_config: "checkpoints/mmpose_config.py"
  pose_checkpoint: "checkpoints/pose.pth"

processing:
  device: "cuda:0"
  detection_threshold: 0.3
  nms_threshold: 0.3
  kpt_threshold: 0.3

noise:
  poisson_scale: 1.2
  gaussian_std: 7.0
  motion_blur_kernel_size: 3
  apply_brightness_reduction: true
  brightness_factor: 20
  target_resolution: [640, 480]
```

---

## Modules

### Configuration

* **Located in**: `config/`
* Uses `dataclasses` for strongly typed config groups.
* `base.py` contains `get_config()` to safely load and cache configs.
* Each config component is modular: `paths_config.py`, `models_config.py`, etc.

### Pose Estimation

* **Located in**: `pose_estimation/`
* `detector.py`: Uses MMDetection to find human bounding boxes.
* `estimator.py`: Uses MMPose to estimate keypoints from bounding boxes.
* `visualizer.py`: Draws keypoints and skeletons.
* `processors/frame_processor.py`: Integrates detection, estimation, visualization.
* `processors/video_loader.py`: Handles reading/writing video.

### Utilities

* **Located in**: `utils/`
* `video_io.py`: Video file loading, frame iteration.
* `json_io.py`: Saving and organizing keypoints.
* `plotting.py`: Visualize filtering results (used later in filtering module).

### Noise Simulation

* **Located in**: `processing/noise_simulator.py`
* Adds Poisson, Gaussian, and motion blur noise.
* Optionally lowers brightness and resizes frames.
* Controlled by `noise` block in YAML.

---
Certainly! Below is the **modified README** with a detailed section added for the `main.py` script, describing its role and how it interacts with CLI commands and the configuration system.

---

## Main Script

### `main.py`

**Purpose**: Acts as the central CLI entry point for running pipeline commands.

**Responsibilities**:

* Parses CLI arguments using `cli.py`.
* Loads the YAML configuration file using `get_config()` from `config/base.py`.
* Dispatches commands to appropriate handler functions defined in `main_handlers.py`.
* Manages logging and error handling centrally.

### Highlights:

* Loads configuration from a YAML file via:

  ```python
  config = get_config(config_path)
  ```
* Handles commands like:

  ```bash
  python main.py detect
  python main.py noise
  ```
* Automatically routes to the correct function (e.g., `run_detection_pipeline`, `simulate_noise`) using:

  ```python
  args.func(args, config)
  ```

### Why Use `main.py`?

* It unifies configuration loading, CLI command handling, and runtime logic.
* Supports scalable extension: You can add new CLI commands like `filter` and `assess` by:

  1. Registering them in `cli.py`
  2. Writing logic in `main_handlers.py`
  3. Calling the corresponding function from your processing module (e.g., `run_filter_pipeline`)

---

## Usage

### CLI Commands

Run commands using:

```bash
python main.py [command] [--flags]
```

#### `detect`

```bash
python main.py detect --video_folder ./videos --output_dir ./results
```

Runs detection + pose estimation + visualization.

#### `noise`

```bash
python main.py noise --input_folder ./videos --output_folder ./noisy_videos
```

Applies simulated noise to video.

#### `filter` *(coming soon)*

Will apply filtering methods to keypoints (e.g., Butterworth, Kalman).

#### `assess` *(coming soon)*

Will compute pose quality metrics like PCK, L2 distance, etc.

---

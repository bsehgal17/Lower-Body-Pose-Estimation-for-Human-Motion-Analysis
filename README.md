# Lower-Body Pose Estimation Pipeline

This project provides a modular and extensible pipeline for performing lower-body human pose estimation from videos. It supports configurable detection, pose estimation, noise simulation, and is designed for reproducibility and research-grade workflows.

The system supports:

* Command-line interface (CLI) for single-step runs.
* A YAML-driven orchestrator (`main.py`) for defining and running multi-step pipelines.

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
* [Pipeline Runner (Single Step)](#pipeline-runner-single-step)
* [Main Orchestrator (Multi-Step)](#main-orchestrator-multi-step)
* [Usage](#usage)

---

## Overview

This pipeline enables lower-body pose analysis on video data through:

* Person detection (using MMDetection)
* Pose keypoint estimation (using MMPose)
* Output of annotated videos and JSON keypoint data
* Optional simulation of degraded video quality for robustness testing

Configuration is handled entirely through YAML files to ensure repeatability and clarity across experiments.

---

## Project Structure

```
project_root/
├── config/                  # Configuration dataclasses and loader
├── pose_estimation/        # Pose estimation logic
│   ├── detector.py
│   ├── estimator.py
│   ├── visualizer.py
│   ├── processors/
│   │   ├── frame_processor.py
│   │   └── video_loader.py
├── noise/
│   └── noise_simulator.py  # Add camera noise, motion blur
├── utils/
│   ├── video_io.py
│   ├── json_io.py
│   └── plotting.py
├── cli.py                  # CLI parser and subcommands
├── main_handlers.py        # Subcommand logic dispatcher
├── pipeline_runner.py      # Run a single processing step
├── main.py                 # Runs a full multi-step pipeline
├── pipelines.yaml          # YAML defining pipeline steps
└── configs_yaml/           # Sample config YAMLs
```

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/lower-body-pose-estimation.git
cd lower-body-pose-estimation
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Install MMPose and MMDetection following their respective documentation.

---

## Configuration Guidelines

Each processing step is configured through a YAML file. This defines models, thresholds, noise parameters, and file paths.

### CLI Overrides (Optional)

While configuration is encouraged via YAML, the following flags can override certain values:

| Command  | CLI Flag                                             | Overrides                    |
| -------- | ---------------------------------------------------- | ---------------------------- |
| `detect` | `--video_folder`, `--output_dir`, `--config_file`    | Paths for input/output       |
| `noise`  | `--input_folder`, `--output_folder`, `--config_file` | Paths for noisy video        |
| *All*    | `--config_file`                                      | Use specific YAML for config |

---

### 1. Pose Estimation

```yaml
models:
  det_config: "checkpoints/det_config.py"
  det_checkpoint: "checkpoints/detector.pth"
  pose_config: "checkpoints/pose_config.py"
  pose_checkpoint: "checkpoints/pose.pth"

processing:
  device: "cuda:0"
  detection_threshold: 0.3
  nms_threshold: 0.3
  kpt_threshold: 0.3
```

### 2. Noise Simulation

```yaml
noise:
  poisson_scale: 1.0
  gaussian_std: 5.0
  motion_blur_kernel_size: 5
  apply_brightness_reduction: true
  brightness_factor: 30
  target_resolution: [1280, 720]
```

### 3. Input/Output Paths

```yaml
paths:
  video_folder: "./videos/input"
  output_dir: "./videos/output"
  csv_file: "./metadata.csv"  # Optional
```

### 4. Video Handling

```yaml
video:
  extensions: [".mp4", ".avi", ".mov"]
```

---

## Modules

### Configuration (`config/`)

* Typed dataclasses ensure safe loading of YAML values.
* `get_config()` validates structure and paths.
* Uses dataclasses for strongly typed config groups.
* base.py contains get_config() to safely load and cache configs.
* Each config component is modular: paths_config.py, models_config.py, etc.

### Pose Estimation (`pose_estimation/`)

* `detector.py`: Uses MMDetection for bounding box prediction.
* `estimator.py`: Uses MMPose for keypoint estimation.
* `visualizer.py`: Overlays pose on video frames.
* `processors/`: Contains video loading and per-frame pipelines.

### Utilities (`utils/`)

* `video_io.py`: Load and iterate over video files.
* `json_io.py`: Store and export keypoint JSONs.
* `plotting.py`: Visualization utilities (for filtering/debugging).

### Noise Simulation (`noise/`)

* `noise_simulator.py`: Adds motion blur, Poisson/Gaussian noise, and brightness reduction.

---

## Pipeline Runner (Single Step)

Use `pipeline_runner.py` to run one processing stage using a config YAML.

```bash
python pipeline_runner.py detect --config_file configs/detect_config.yaml
```

Supported steps:

* `detect`
* `noise`
* `filter` *(coming soon)*
* `assess` *(coming soon)*

---

## Main Orchestrator (Multi-Step)

Use `main.py` to run a sequence of pipeline steps defined in a single YAML file (`pipelines.yaml`).

### Example: `pipelines.yaml`

```yaml
orchestrator:
  log_dir: "./logs"
  default_device: "cuda:0"

pipelines:
  - name: run_pose_estimation
    steps:
      - command: detect
        config_file: "configs/detect_config.yaml"

  - name: simulate_noise
    steps:
      - command: noise
        config_file: "configs/noise_config.yaml"

  - name: rerun_on_noisy
    steps:
      - command: detect
        config_file: "configs/detect_on_noisy.yaml"
```

### Run all steps

```bash
python main.py
```

This will internally execute:

```bash
python pipeline_runner.py detect --config_file configs/detect_config.yaml
python pipeline_runner.py noise --config_file configs/noise_config.yaml
python pipeline_runner.py detect --config_file configs/detect_on_noisy.yaml
```

Each step is run via subprocess with live logging.

---

## Usage

### CLI (Single Step)

```bash
python pipeline_runner.py [command] --config_file path/to/config.yaml
```

#### `detect`

Run detection + pose estimation + visualization:

```bash
python pipeline_runner.py detect --config_file configs/detect_config.yaml
```

#### `noise`

Apply noise simulation to clean videos:

```bash
python pipeline_runner.py noise --config_file configs/noise_config.yaml
```

#### `filter` *(coming soon)*

#### `assess` *(coming soon)*

---


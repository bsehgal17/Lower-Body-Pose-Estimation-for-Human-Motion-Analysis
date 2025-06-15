# Lower-Body Pose Estimation Pipeline

This project provides a **modular, extensible, and dataset-aware** pipeline for performing lower-body human pose estimation from videos. It supports configurable detection, pose estimation, filtering, noise simulation, and quantitative assessment of accuracy using evaluation metrics. The system is designed for **reproducibility, research workflows**, and robust testing via simulation.

---

## Table of Contents

* [Overview](#overview)
* [Project Structure](#project-structure)
* [Installation](#installation)
* [Configuration Guidelines](#configuration-guidelines)

  * [1. Pose Estimation](#1-pose-estimation)
  * [2. Noise Simulation](#2-noise-simulation)
  * [3. Filtering](#3-filtering)
  * [4. Evaluation](#4-evaluation)
  * [5. Input/Output Paths](#5-inputoutput-paths)
* [Pipeline Runner (Single Step)](#pipeline-runner-single-step)
* [Main Orchestrator (Multi-Step)](#main-orchestrator-multi-step)
* [Usage](#usage)

---

## Overview

This pipeline enables lower-body pose analysis from video files using:

* **Detection** with MMDetection
* **Pose Estimation** with MMPose
* **Optional filtering** of noisy or raw keypoints
* **Noise simulation** to degrade videos for robustness testing
* **Metric-based evaluation** comparing predictions with ground truth

All logic is **fully configurable via YAML**, with clean separation of stages and built-in dataset support (e.g., HumanEva).

---

## Project Structure

```
project_root/
├── config/                  # Configuration dataclasses and loader
│   ├── pipeline_config.py
│   ├── global_config.py
│   ├── dataset_config.py
├── pose_estimation/        # Pose estimation logic
│   ├── detector.py
│   ├── estimator.py
│   ├── visualizer.py
├── filtering_and_data_cleaning/
│   ├── filter_registry.py
│   ├── preprocessing_utils.py
├── evaluation/             # Evaluation metric logic
│   ├── overall_pck.py
│   ├── jointwise_pck.py
│   ├── evaluation_registry.py
├── dataset_files/
│   └── HumanEva/
│       ├── humaneva_evaluation.py
│       ├── get_gt_keypoint.py
│       └── humaneva_metadata.py
├── noise/
│   └── noise_simulator.py
├── utils/
│   ├── video_io.py
│   ├── extract_predicted_points.py
│   ├── import_utils.py
├── cli.py                  # CLI parser and subcommands
├── main_handlers.py        # Subcommand logic dispatcher
├── pipeline_runner.py      # Run a single processing step
├── main.py                 # Runs a full multi-step pipeline
├── pipelines.yaml          # Multi-step orchestrator YAML
└── config_yamls/           # Sample YAML configurations
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

3. Follow official installation guides to install:

* [MMPose](https://github.com/open-mmlab/mmpose)
* [MMDetection](https://github.com/open-mmlab/mmdetection)

Ensure they are added to your `PYTHONPATH` if necessary.

---

## Configuration Guidelines

Each step in the pipeline is driven by YAML files with typed dataclass validation. Below are example configs and expected fields.

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

### 3. Filtering

```yaml
filter:
  name: "butterworth"
  params:
    cutoff: 0.1
    order: 3
  enable_interpolation: true
  interpolation_kind: "linear"
  outlier_removal:
    enable: true
    method: "iqr"
    params:
      threshold: 1.5
  joints_to_filter: ["LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_HIP", "RIGHT_HIP"]
  enable_filter_plots: true
```

### 4. Evaluation

```yaml
evaluation:
  input_dir: "results/detect_filtered"
  metrics:
    - name: overall_pck
      params:
        threshold: 0.05
    - name: jointwise_pck
      params:
        threshold: 0.05
```

Also ensure:

```yaml
dataset:
  joint_enum_module: utils.joint_enum.GTJointsHumanEVa
  keypoint_format: utils.joint_enum.PredJointsCOCOWholebody
  sync_data:
    data:
      S1:
        Walking 1: [667, 667, 667]
```

### 5. Input/Output Paths

```yaml
paths:
  dataset: "HumanEva"
  ground_truth_file: "data/ground_truth.csv"
  input_dir: "data/raw"
  output_dir: "results"
```

---

## Pipeline Runner (Single Step)

To run a single processing step:

```bash
python pipeline_runner.py detect --config_file config_yamls/detect_config.yaml
```

Supported commands:

* `detect`: Run detection + pose estimation
* `noise`: Apply video degradation
* `filter`: Apply time-series filters to keypoints
* `assess`: Evaluate predictions using ground truth

Each step stores output in a structured folder under `output_dir`.

---

## Main Orchestrator (Multi-Step)

Run an entire pipeline (e.g., detect + filter + assess) from a single YAML:

```yaml
orchestrator:
  log_dir: "./logs"
  default_device: "cuda:0"

pipelines:
  - name: full_pipeline
    steps:
      - command: detect
        config_file: "config_yamls/detect_config.yaml"
      - command: filter
        config_file: "config_yamls/filter_config.yaml"
      - command: assess
        config_file: "config_yamls/eval_config.yaml"
```

Run it with:

```bash
python main.py
```

Each step executes in isolation and stores intermediate outputs automatically.

---

## Usage

### CLI (Single Step)

```bash
python pipeline_runner.py [command] --config_file path/to/config.yaml
```

#### `detect`

```bash
python pipeline_runner.py detect --config_file config_yamls/detect_config.yaml
```

#### `noise`

```bash
python pipeline_runner.py noise --config_file config_yamls/noise_config.yaml
```

#### `filter`

```bash
python pipeline_runner.py filter --config_file config_yamls/filter_config.yaml
```

#### `assess`

```bash
python pipeline_runner.py assess --config_file config_yamls/eval_config.yaml
```

---

## Output Structure

Each pipeline stage writes its outputs to `output_dir/pipeline_name/step_name/`. For example:

```
results/
├── my_pipeline/
│   ├── detect/
│   ├── filter/
│   └── assess/
```

Evaluation will create a combined Excel file like:

```bash
degraded_videos_overall_pck.xlsx
```

Containing metric-wise scores across all evaluated videos.

---

## Notes

* Enum class names (e.g. `GTJointsHumanEVa`) are dynamically imported from strings in the config.
* The evaluation system supports multiple metrics and aggregates into a single Excel output.

---

This modular setup is ideal for research and benchmarking. Let us know if you'd like to extend support for new datasets or add custom metrics!

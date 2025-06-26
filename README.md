# Lower-Body Pose Estimation Pipeline

A **modular, extensible, and dataset-aware** pipeline for lower-body human pose estimation from videos. Designed for **reproducible research**, robust testing, and easy benchmarking.

---

## 🚀 Features

* **Detection** (MMDetection) and **Pose Estimation** (MMPose)
* **Noise Simulation** for robustness testing
* **Filtering** and cleaning of keypoints
* **Metric-based Evaluation** (e.g., PCK)
* **YAML-based configuration** for all steps
* **Dataset support** (e.g., HumanEva)
* **CLI and Orchestrator** for single or multi-step workflows
* **Global vs Pipeline-specific config** for flexible setup

---

## 🗂️ Project Structure

<details>
<summary>Click to expand</summary>

```
project_root/
├── config/                  # Config dataclasses and loader
├── pose_estimation/         # Detection & pose estimation
├── filtering_and_data_cleaning/
├── evaluation/              # Metrics & evaluation
├── dataset_files/
│   └── HumanEva/
├── noise/
├── utils/
├── cli.py                   # CLI parser
├── main_handlers.py         # Subcommand dispatcher
├── pipeline_runner.py       # Single-step runner
├── main.py                  # Multi-step orchestrator
├── pipelines.yaml           # Example orchestrator config
├── config_yamls/
│   ├── global_config.yaml   # Shared config across all pipelines
│   └── <pipeline>.yaml      # Step-specific configs
```

</details>

---

## 📦 Installation

1. Clone the repository:
```
bash
git clone https://github.com/yourusername/lower-body-pose-estimation.git
cd lower-body-pose-estimation
```

2. Install dependencies:
```
bash
pip install -r requirements.txt

```
3. Follow official installation guides to install:

* [MMPose](https://github.com/open-mmlab/mmpose)
* [MMDetection](https://github.com/open-mmlab/mmdetection)

Ensure they are added to your PYTHONPATH if necessary.

---
## ⚙️ Configuration Guidelines

### 🔁 Config Files

The system uses **two layers of configuration**:

#### 1. Global Config (`global_config.yaml`)

Shared settings applicable across all pipelines and datasets.

```yaml
paths:
  dataset_root: "data/raw"
  results_root: "results"

video:
  extensions: [".mp4", ".avi"]
```

#### 2. Pipeline Config (`<pipeline>_config.yaml`)

Step-specific logic and hyperparameters (e.g., filtering, noise simulation, metric selection).
Specific to each pipeline step or orchestrated workflow.

Includes dataset-aware logic and settings for detection, filtering, etc.

---

### 1. Pose Estimation
```
yaml
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

---

### 2. Noise Simulation
```
yaml
noise:
  poisson_scale: 1.0
  gaussian_std: 5.0
  motion_blur_kernel_size: 5
  apply_brightness_reduction: true
  brightness_factor: 30
  target_resolution: [1280, 720]

```
---

### 3. Filtering
```
yaml
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
---

### 4. Evaluation
```
yaml
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
#### Dataset-Specific Parameters in Pipeline Config
```
yaml
dataset:
  name: "HumanEva"
  joint_enum_module: utils.joint_enum.GTJointsHumanEVa
  keypoint_format: utils.joint_enum.PredJointsCOCOWholebody
  sync_data:
    data:
      S1:
        Walking 1: [667, 667, 667]

```
---

## 🔄 Input Directory Resolution for `noise` and `filter`

For **multi-step pipelines**, the system resolves the `input_dir` automatically when not explicitly provided:

### 📥 `noise` step:

* Uses `noise.input_dir` if explicitly set.
* Otherwise falls back to the **detection output folder**, e.g. `.../detect/`.

### 🧼 `filter` step:

* Uses `filter.input_dir` if explicitly set.
* Otherwise:

  * If a `noise` step has already run, it uses `noise.output_dir`.
  * If not, it uses `detect.output_dir`.

This behavior ensures that `noise → filter` pipelines require **minimal configuration** while remaining reproducible and extensible.

---

## 🧠 Filter Parameter Options

The `filter.params` section supports multiple formats for flexible testing and hyperparameter sweeps:

| Format               | Example                  | Behavior                   |
| -------------------- | ------------------------ | -------------------------- |
| **Single value**     | `sigma: 1.0`             | Runs once with that value  |
| **List of values**   | `sigma: [0.5, 1.0, 1.5]` | Runs once per listed value |
| **Range expression** | `sigma: "range(1, 4)"`   | Interpreted as \[1, 2, 3]  |

Each unique parameter combination results in a **separate filtered output**, saved to a subfolder like:

```
.../filter/gaussian_sigma1.0/
.../filter/gaussian_sigma1.5/
```

---

### Example Filter Config

```yaml
filter:
  name: "gaussian"
  params:
    sigma: "range(1, 4)"  # expands to [1, 2, 3]

  enable_interpolation: true
  interpolation_kind: "linear"

  outlier_removal:
    enable: true
    method: "iqr"
    params:
      iqr_multiplier: 1.5

  joints_to_filter:
    - LEFT_ANKLE
    - RIGHT_ANKLE
    - LEFT_KNEE
    - RIGHT_KNEE
    - LEFT_HIP
    - RIGHT_HIP

  enable_filter_plots: true
```

---

## 🛠️ Pipeline Runner (Single Step)

To run a single processing step:

```bash
python pipeline_runner.py filter --config_file config_yamls/filter_config.yaml
```

Supported commands:

* `detect`: Run detection + pose estimation
* `noise`: Apply video degradation
* `filter`: Apply time-series filters to keypoints
* `assess`: Evaluate predictions using ground truth

---

## 🔀 Main Orchestrator (Multi-Step)

Define a full pipeline in `pipelines.yaml`:

```yaml
orchestrator:
  log_dir: "./logs"
  default_device: "cuda:0"

pipelines:
  - name: run_pose_estimation_and_filter
    steps:
      - command: detect
        config_file: "config_yamls/detect_config.yaml"
      - command: noise
        config_file: "config_yamls/noise_config.yaml"
      - command: filter
        config_file: "config_yamls/filter_config.yaml"
```

Run it with:

```bash
python main.py
```
---

## 📌 Usage

### CLI (Single Step)
```
bash
python pipeline_runner.py [command] --config_file path/to/config.yaml

```
Examples:
```
bash
python pipeline_runner.py detect --config_file config_yamls/detect_config.yaml
python pipeline_runner.py noise --config_file config_yamls/noise_config.yaml
python pipeline_runner.py filter --config_file config_yamls/filter_config.yaml
python pipeline_runner.py assess --config_file config_yamls/eval_config.yaml

```

---

## 📂 Output Structure

Each stage stores its results in:

```
results/
└── run_pose_estimation_and_filter/
    ├── detect/
    ├── noise/
    └── filter/
        ├── gaussian_sigma1/
        └── butterworth_window5_order2/
```

---

## 📝 Notes

* `noise` and `filter` automatically resolve their inputs unless overridden.
* Parameters for filtering can be expanded via lists or range expressions.
* Config structure supports modular extension for new datasets, models, or metrics.
* Filtered keypoints are saved in both `.json` and `.pkl` formats.
* Global vs Pipeline configs cleanly separate common settings from pipeline-specific ones.
* Dataset-specific logic (joint enums, keypoint formats, sync info) should be passed via the pipeline config.
  
---

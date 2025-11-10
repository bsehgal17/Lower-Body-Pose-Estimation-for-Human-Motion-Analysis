# HumanSC3D Lower Body Pipeline Configuration Guide

## Overview
This guide explains how to use the HumanSC3D lower body pose estimation configurations for detection and evaluation.

## Configuration Files Created

### Main Pipeline Configurations
1. **`humansc3d_lowerbody_pipeline.yaml`** - Complete pipeline orchestrator
2. **`humansc3d_lowerbody_detect_eval.yaml`** - Combined detection and evaluation config

### Detector Configurations
1. **`rtmw_detector_humansc3d.yaml`** - RTMW detector for HumanSC3D
2. **`mediapipe_detector_humansc3d.yaml`** - MediaPipe detector for HumanSC3D

### Evaluation-Only Configurations
1. **`humansc3d_config_nojoints.yaml`** - RTMW evaluation without joint filtering
2. **`humansc3d_config_mediapipe_nojoints.yaml`** - MediaPipe evaluation without joint filtering

## Usage Examples

### 1. Full Lower Body Pipeline (Detection + Evaluation)
```bash
# Run complete pipeline with RTMW
python pipeline_runner.py --config config_yamls/humansc3d_lowerbody_pipeline.yaml

# Or run specific pipeline
python main.py --config dataset_files/HumanSC3D/humansc3d_lowerbody_detect_eval.yaml
```

### 2. Detection Only
```bash
# RTMW detection
python main.py --config config_yamls/detector/rtmw_detector_humansc3d.yaml --command detect

# MediaPipe detection
python main.py --config config_yamls/detector/mediapipe_detector_humansc3d.yaml --command detect
```

### 3. Evaluation Only (using pre-computed detections)
```bash
# RTMW evaluation
python main.py --config dataset_files/HumanSC3D/humansc3d_config_nojoints.yaml --command evaluation

# MediaPipe evaluation
python main.py --config dataset_files/HumanSC3D/humansc3d_config_mediapipe_nojoints.yaml --command evaluation
```

### 4. Individual Pipeline Steps
```bash
# Detection only
python pipeline_runner.py --config config_yamls/humansc3d_lowerbody_pipeline.yaml --pipeline HumanSC3D_Detection_Only

# Evaluation only
python pipeline_runner.py --config config_yamls/humansc3d_lowerbody_pipeline.yaml --pipeline HumanSC3D_Evaluation_Only

# Filtering only
python pipeline_runner.py --config config_yamls/humansc3d_lowerbody_pipeline.yaml --pipeline HumanSC3D_Filter_Only
```

## Lower Body Focus

All configurations are specifically optimized for lower body joints:
- **LEFT_HIP**
- **RIGHT_HIP**
- **LEFT_KNEE**
- **RIGHT_KNEE**
- **LEFT_ANKLE**
- **RIGHT_ANKLE**

## Evaluation Metrics

The configurations include comprehensive lower body evaluation:

### PCK (Percentage of Correct Keypoints)
- PCK@0.02 and PCK@0.04 thresholds
- Overall, per-frame, and joint-wise analysis
- Focused on lower body joints only

### Metrics Computed
1. **overall_pck** - Overall accuracy across all lower body joints
2. **per_frame_pck** - Frame-by-frame temporal analysis
3. **jointwise_pck** - Individual joint performance analysis

## Dataset Configuration

### Path Structure
- **Videos**: `{subject}/videos/{camera}/{action}.mp4`
- **Ground Truth**: `{subject}/processed_outputs/2d_points/{action}_{camera}_2d.json`

### Supported Subjects
- s01, s02, s03, s04, s05, s06, s07, s08, s09, s10

## Key Features

### Detection
- ✅ RTMW pose estimation model
- ✅ MediaPipe pose estimation
- ✅ Confidence filtering for multiple skeletons
- ✅ Noise simulation (Gaussian, Poisson, motion blur, brightness)

### Filtering
- ✅ Gaussian smoothing
- ✅ Outlier removal (IQR method)
- ✅ Linear interpolation for missing frames
- ✅ Visualization plots

### Evaluation
- ✅ Lower body joint focus
- ✅ Multiple PCK thresholds
- ✅ Temporal and spatial analysis
- ✅ Ground truth comparison

## File Locations

```
config_yamls/
├── humansc3d_lowerbody_pipeline.yaml          # Main pipeline orchestrator
├── pipelines_detection.yaml                    # Updated with HumanSC3D entries
└── detector/
    ├── rtmw_detector_humansc3d.yaml            # RTMW detector config
    └── mediapipe_detector_humansc3d.yaml       # MediaPipe detector config

dataset_files/HumanSC3D/
├── humansc3d_lowerbody_detect_eval.yaml       # Combined detect+eval config
├── humansc3d_config_nojoints.yaml             # RTMW evaluation only
├── humansc3d_config_mediapipe_nojoints.yaml   # MediaPipe evaluation only
└── ... (other existing files)
```

## Quick Start

1. **Update paths in configurations** to match your system
2. **Run detection and evaluation**:
   ```bash
   python pipeline_runner.py --config config_yamls/humansc3d_lowerbody_pipeline.yaml --pipeline HumanSC3D_LowerBody_RTMW
   ```
3. **Check results** in the output directory specified in the logs

## Troubleshooting

- Ensure all path configurations match your system setup
- Verify CUDA availability for RTMW (or use CPU for MediaPipe)
- Check ground truth file formats match expected JSON structure
- Adjust confidence thresholds if getting too few/many detections
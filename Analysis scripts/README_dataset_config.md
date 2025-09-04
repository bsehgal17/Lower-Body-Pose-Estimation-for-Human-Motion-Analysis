# Dataset Configuration Documentation

This document explains how to configure datasets using YAML files instead of hardcoded Python configurations.

## Overview

Dataset configurations are now stored in YAML files located in the `config_yamls/` directory:
- `humaneva_config.yaml` - Configuration for HumanEva dataset
- `movi_config.yaml` - Configuration for MoVi dataset

## YAML Configuration Structure

### Basic Structure
```yaml
dataset:
  name: "dataset_name"
  model: "model_name"

paths:
  video_directory: "/path/to/videos"
  pck_file_path: "/path/to/pck/metrics.xlsx"
  save_folder: "/path/to/save/results"

columns:
  subject_column: "subject"
  action_column: "action"  # or null if not applicable
  camera_column: "camera"  # or null if not applicable

pck_scores:
  overall:
    - "overall_overall_pck_0.10"
    - "overall_overall_pck_0.20"
  
  per_frame:
    - "pck_per_frame_pck_0.10"
    - "pck_per_frame_pck_0.20"

sync_data:  # or null if not applicable
  S1:
    "Walking 1": [667, 667, 667]
    "Jog 1": [49, 50, 51]
```

## How to Modify Configurations

### 1. Change File Paths
Edit the `paths` section in the YAML file:
```yaml
paths:
  video_directory: "/your/new/video/path"
  pck_file_path: "/your/new/pck/file.xlsx"
  save_folder: "/your/new/save/folder"
```

### 2. Add/Remove PCK Score Columns
Modify the `pck_scores` section:
```yaml
pck_scores:
  overall:
    - "overall_overall_pck_0.05"  # Add new threshold
    - "overall_overall_pck_0.10"
    - "overall_overall_pck_0.20"
    - "overall_overall_pck_0.50"
  
  per_frame:
    - "pck_per_frame_pck_0.05"    # Add corresponding per-frame column
    - "pck_per_frame_pck_0.10"
    - "pck_per_frame_pck_0.20"
    - "pck_per_frame_pck_0.50"
```

### 3. Update Synchronization Data
For datasets that require frame synchronization:
```yaml
sync_data:
  S1:
    "Walking 1": [667, 667, 667]
    "Running 1": [100, 101, 102]  # Add new action
  S4:  # Add new subject
    "Walking 1": [500, 500, 500]
    "Jog 1": [200, 201, 202]
```

### 4. Change Dataset Metadata
```yaml
dataset:
  name: "new_dataset"
  model: "DWPose"  # Change model type

columns:
  subject_column: "participant"  # Different column name
  action_column: "activity"
  camera_column: "view"
```

## Adding New Datasets

To add a new dataset:

1. Create a new YAML file: `config_yamls/newdataset_config.yaml`
2. Follow the structure above
3. Update `config_factory.py` to recognize the new dataset:

```python
elif dataset_name == "newdataset":
    return ConfigFactory._create_config_from_yaml("newdataset")
```

## Testing Configurations

Test your YAML configurations:
```bash
cd "Analysis scripts"
python test_yaml_config.py
```

This will validate all dataset configurations and show any errors.

## Benefits of YAML Configuration

1. **Easy to Edit**: No need to modify Python code
2. **Version Control Friendly**: Clear diffs when paths change
3. **Environment Specific**: Different YAML files for different environments
4. **Validation**: Automatic validation of required fields
5. **Documentation**: Self-documenting configuration structure

## Migration from Hardcoded Config

The old hardcoded configurations are still available as fallback but will show deprecation warnings. It's recommended to update all configurations to use YAML files.

## Example: Switching Environments

Development environment (`config_yamls/movi_config_dev.yaml`):
```yaml
paths:
  video_directory: "/local/dev/videos"
  save_folder: "/local/dev/results"
```

Production environment (`config_yamls/movi_config_prod.yaml`):
```yaml
paths:
  video_directory: "/storage/prod/videos"
  save_folder: "/storage/prod/results"
```

Then modify the config factory to load different files based on environment variables.

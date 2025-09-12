# HumanEva Dataset Ground Truth Setup Guide

This comprehensive guide explains how to obtain, process, and prepare the **HumanEva** dataset ground truth for use in our lower-body pose estimation pipeline. The HumanEva dataset provides synchronized motion capture data and multi-view video recordings of human activities.

## Table of Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Dataset Structure](#dataset-structure)
- [Step-by-Step Setup](#step-by-step-setup)
- [Understanding Output Files](#understanding-output-files)
- [Synchronization and Validation](#synchronization-and-validation)
- [Joint Mapping](#joint-mapping)
- [Integration with Pipeline](#integration-with-pipeline)
- [Troubleshooting](#troubleshooting)
- [Advanced Features](#advanced-features)

---

## Overview

The HumanEva ground truth setup process involves:
1. **Data Acquisition**: Downloading preprocessed `.npz` files from VideoPose3D
2. **Format Conversion**: Converting NPZ → NPY → CSV for structured access
3. **Data Filtering**: Extracting only the validated `chunk0` segments
4. **Synchronization**: Aligning motion capture data with video frames
5. **Visualization**: Generating overlay videos for quality verification

### Key Features
- Automated NPZ to CSV conversion pipeline
- Frame-accurate video-mocap synchronization
- Multi-camera support (C1, C2, C3)
- Quality control through visualization overlays
- Integration with evaluation metrics (PCK, mAP)

---

## Prerequisites

### Required Files
Download the HumanEva dataset with the following structure:

```
HumanEva/
├── S1/                           # Subject 1 data
│   └── Image_Data/
│       ├── Walking_1_(C1).avi
│       ├── Walking_1_(C2).avi
│       ├── Walking_1_(C3).avi
│       ├── Jog_1_(C1).avi
│       └── ...
├── S2/                           # Subject 2 data
│   └── Image_Data/
│       └── ...
├── S3/                           # Subject 3 data
│   └── Image_Data/
│       └── ...
└── data_2d_humaneva20_gt.npz     # Ground truth from VideoPose3D
```

### Dependencies
```bash
pip install opencv-python numpy pandas matplotlib scipy
```

---

## Dataset Structure

### Input File Types

| File Type | Description | Contains |
|-----------|-------------|----------|
| `data_2d_humaneva20_gt.npz` | VideoPose3D ground truth | 2D keypoints for all subjects/actions |
| `Walking_1_(C1).avi` | Video recordings | Multi-camera view sequences |
| `Jog_1_(C1).avi` | Video recordings | Action-specific video segments |

### Ground Truth Data Structure
The NPZ file contains a hierarchical dictionary:
```python
{
  "S1": {
    "Walking 1 chunk0": [camera1_data, camera2_data, camera3_data],
    "Jog 1 chunk0": [camera1_data, camera2_data, camera3_data]
  },
  "S2": { ... },
  "S3": { ... }
}
```

Each camera data array has shape: `(frames, joints, 2)` for 2D coordinates.

---

## Step-by-Step Setup

### Step 1: Obtain Ground Truth (NPZ Format)

Download the preprocessed ground truth from [VideoPose3D](https://github.com/facebookresearch/VideoPose3D):

```bash
# Follow VideoPose3D's DATASETS.md guide
wget https://dl.fbaipublicfiles.com/video-pose-3d/data_2d_humaneva20_gt.npz
```

This file contains:
- All 2D joint positions projected to image space
- Grouped by subject (S1, S2, S3), action (Walking, Jog), and camera (C1, C2, C3)
- Motion capture data aligned with video sequences

### Step 2: Convert NPZ to Individual CSV Files

Use the `npy_to_csv.py` script to convert the NPZ data into structured CSV files:

```python
# Configure paths in npy_to_csv.py
npz_file = r"C:\path\to\data_2d_humaneva20_gt.npz"
output_folder = r"C:\path\to\output_csvs"

# Run conversion
python npy_to_csv.py
```

**What this script does:**
- Extracts the nested dictionary structure from the NPZ file
- Converts each `(subject, action, camera)` combination to a separate CSV
- Generates columns: `Subject`, `Action`, `Camera`, `x1`, `y1`, ..., `x20`, `y20`
- Creates organized folder structure for easy navigation

**Output Structure:**
```
output_csvs/
├── S1/
│   ├── Walking_1_chunk0.csv
│   ├── Jog_1_chunk0.csv
│   └── ...
├── S2/
│   └── ...
└── S3/
    └── ...
```

**Key Features of npy_to_csv.py:**
- Handles multi-dimensional arrays automatically
- Supports both single and multi-camera data
- Flattens keypoint coordinates into readable CSV format
- Adds metadata columns for easy filtering

### Step 3: Combine CSV Files for Validation

The HumanEva dataset provides action sequences divided into chunks (`chunk0`, `chunk1`, etc.). Following VideoPose3D evaluation standards, we use **only `chunk0`** segments, which represent the most validated and reliable motion capture data.

Use `get_combined_csv.py` to create a unified ground truth file:

```python
# Configure paths in get_combined_csv.py
combine_all_validate_csvs(
    input_dir=r"C:\path\to\output_csvs",
    output_path=r"C:\path\to\validate_combined_chunk0.csv"
)
```

**What this script does:**
- Scans all generated CSV files in directories containing 'Validate'
- Filters for files containing `chunk0` in the filename
- Cleans the `Subject` column format (removes path artifacts)
- Adds an `action_group` column by removing 'chunk0' from 'Action'
- Combines all data into a single reference file

**Key Processing Steps:**
1. **Path Cleaning**: Removes directory separators from Subject names
2. **Action Grouping**: Creates `action_group` field for easier filtering
3. **Data Validation**: Ensures consistent column structure across files
4. **Error Handling**: Gracefully handles corrupted or malformed CSV files

### Step 4: Generate Visualization Overlays

Use `get_sync_data.py` to create video overlays with ground truth keypoints:

```python
# Configure paths in get_sync_data.py
overlay_gt_on_videos(
    video_root=r"C:\path\to\HumanEva",
    csv_file=r"C:\path\to\validate_combined_chunk0.csv",
    output_root=r"C:\path\to\output_overlays"
)
```

**What this script does:**
- Parses video file names to extract metadata (subject, action, camera)
- Matches video files with corresponding ground truth data
- Overlays green circles on joint positions
- Preserves original video quality and frame rate
- Creates organized output structure matching input hierarchy

**Features:**
- **Metadata Parsing**: Automatically extracts subject, action, and camera from filenames
- **CSV Matching**: Finds corresponding ground truth data for each video
- **Quality Control**: Only processes videos with matching ground truth data
- **Batch Processing**: Handles entire directory structures recursively

---

## Understanding Output Files

### File Formats

#### 1. Individual CSV Files (`{Subject}/{Action}_chunk0.csv`)
- **Format**: Structured CSV with flat keypoint layout
- **Columns**: `Subject`, `Action`, `Camera`, `x1`, `y1`, ..., `x20`, `y20`
- **Rows**: Each row represents one frame for a specific camera view
- **Usage**: Subject/action-specific analysis and debugging

**Example structure:**
```csv
Subject,Action,Camera,x1,y1,x2,y2,...,x20,y20
S1,Walking 1 chunk0,1,156.2,234.7,167.8,245.3,...,189.4,298.1
S1,Walking 1 chunk0,1,157.1,235.2,168.2,246.1,...,190.2,299.3
```

#### 2. Combined CSV File (`validate_combined_chunk0.csv`)
- **Format**: Unified CSV containing all chunk0 data
- **Additional Columns**: `action_group` for easier filtering
- **Structure**: All subjects, actions, and cameras in one file
- **Usage**: Main ground truth reference for evaluation pipeline

**Example structure:**
```csv
Subject,Action,Camera,x1,y1,x2,y2,...,x20,y20,action_group
S1,Walking 1 chunk0,1,156.2,234.7,...,189.4,298.1,Walking 1
S1,Walking 1 chunk0,2,154.8,232.1,...,187.9,296.7,Walking 1
S2,Jog 1 chunk0,1,145.3,221.4,...,178.2,285.6,Jog 1
```

#### 3. Overlay Videos (`{action}_{camera}_overlay.avi`)
- **Format**: Video files with keypoint overlays
- **Content**: Original frames with ground truth joints as green circles
- **Purpose**: Quality control and visual verification
- **Codec**: XVID for cross-platform compatibility

---

## Synchronization and Validation

### Frame Synchronization

HumanEva requires precise synchronization between motion capture data and video frames. The pipeline handles this through:

#### Sync Data Configuration
The `humaneva_config.yaml` contains frame offsets for each video:

```yaml
dataset:
  sync_data:
    data:
      S1:
        Walking 1: [667, 667, 667]  # Frame offsets for C1, C2, C3
        Jog 1: [49, 50, 51]
      S2:
        Walking 1: [547, 547, 546]
        Jog 1: [493, 491, 502]
      S3:
        Walking 1: [524, 524, 524]
        Jog 1: [464, 462, 462]
```

**Why Synchronization is Needed:**
- Motion capture and video recording may have different start times
- Frame rates might not be perfectly aligned
- Hardware latency can introduce temporal offsets

#### Implementation in humaneva_evaluation.py
The evaluation script automatically applies synchronization:

```python
# Extract sync offset for current sample
sync_start = pipeline_config.dataset.sync_data["data"][subject][action][camera_idx]

# Adjust prediction data to align with ground truth
pred_keypoints = pred_keypoints[sync_start:]
pred_bboxes = pred_bboxes[sync_start:]
pred_scores = pred_scores[sync_start:]

# Ensure equal lengths for comparison
min_len = min(len(gt_keypoints), len(pred_keypoints))
```

### Resolution Handling

The pipeline automatically handles resolution differences between original and processed videos:

```python
# Get original and processed video resolutions
orig_w, orig_h = get_video_resolution(original_video_path)
test_w, test_h = get_video_resolution(pred_video_path)

# Rescale keypoints if necessary
if (test_w, test_h) != (orig_w, orig_h):
    scale_x = orig_w / test_w
    scale_y = orig_h / test_h
    
    # Rescale both keypoints and bounding boxes
    for i in range(len(pred_keypoints)):
        if pred_keypoints[i] is not None:
            pred_keypoints[i] = rescale_keypoints(pred_keypoints[i], scale_x, scale_y)
            # Rescale bounding boxes too
            bbox = pred_bboxes[i]
            coords = bbox[0]
            pred_bboxes[i] = [
                coords[0] * scale_x, coords[1] * scale_y,
                coords[2] * scale_x, coords[3] * scale_y
            ]
```

---

## Joint Mapping

### HumanEva Joint Layout (GTJointsHumanEva)

The HumanEva dataset uses a 20-joint skeletal model. The exact joint mapping depends on the VideoPose3D preprocessing, but typically includes:

```python
class GTJointsHumanEva(Enum):
    # Lower Body (Primary Focus)
    LEFT_HIP = 0
    RIGHT_HIP = 1
    LEFT_KNEE = 2
    RIGHT_KNEE = 3
    LEFT_ANKLE = 4
    RIGHT_ANKLE = 5
    LEFT_FOOT = 6
    RIGHT_FOOT = 7
    
    # Upper Body
    HEAD = 8
    NECK = 9
    LEFT_SHOULDER = 10
    RIGHT_SHOULDER = 11
    LEFT_ELBOW = 12
    RIGHT_ELBOW = 13
    LEFT_WRIST = 14
    RIGHT_WRIST = 15
    
    # Torso
    SPINE = 16
    PELVIS = 17
    LEFT_CLAVICLE = 18
    RIGHT_CLAVICLE = 19
```

### Joint Coordinate System
- **Origin**: Top-left corner of image (0,0)
- **X-axis**: Increases rightward
- **Y-axis**: Increases downward
- **Units**: Pixels
- **Range**: [0, image_width] × [0, image_height]

### Important Notes
- Joint indices are **0-based** in the exported arrays
- Lower body joints (hips, knees, ankles) are the primary focus for this pipeline
- All joints are available but evaluation typically focuses on locomotion-relevant joints
- Joint visibility/confidence scores are not included in the HumanEva ground truth

---

## Integration with Pipeline

### Ground Truth Loading

The `GroundTruthLoader` class provides structured access to the data:

```python
from dataset_files.HumanEva.get_gt_keypoint import GroundTruthLoader

# Initialize loader with combined CSV
loader = GroundTruthLoader("path/to/validate_combined_chunk0.csv")

# Extract keypoints for specific sample
keypoints = loader.get_keypoints(
    subject="S1",
    action_group="Walking 1",  # Note: uses action_group, not action
    camera=0,                  # 0-indexed camera ID
    chunk="chunk0"
)

# Returns: numpy array of shape (frames, joints, 2)
print(f"Keypoints shape: {keypoints.shape}")
```

**Key Features of GroundTruthLoader:**
- **Flexible Filtering**: Filter by subject, action, camera, and chunk
- **Auto-detection**: Automatically detects keypoint columns using regex
- **Error Handling**: Provides clear error messages for missing data
- **Data Validation**: Ensures consistent data types and shapes

### Configuration Integration

Update your pipeline configuration (`humaneva_config.yaml`):

```yaml
paths:
  dataset: HumanEva
  ground_truth_file: /path/to/validate_combined_chunk0.csv

dataset:
  joint_enum_module: utils.joint_enum.GTJointsHumanEva
  keypoint_format: utils.joint_enum.PredJointsDeepLabCut
  
  # Frame synchronization data
  sync_data:
    data:
      S1:
        Walking 1: [667, 667, 667]
        Jog 1: [49, 50, 51]

evaluation:
  metrics:
    - name: overall_pck
      params:
        threshold: 0.25
        joints_to_evaluate:
          - LEFT_HIP
          - RIGHT_HIP
          - LEFT_KNEE
          - RIGHT_KNEE
          - LEFT_ANKLE
          - RIGHT_ANKLE
```

### Evaluation Metrics

The pipeline supports comprehensive evaluation metrics:

#### 1. Overall PCK (Percentage of Correct Keypoints)
- **Thresholds**: 0.25, 0.3, 0.35, 0.4
- **Purpose**: Overall pose estimation accuracy
- **Calculation**: Percentage of keypoints within threshold distance

#### 2. Per-frame PCK
- **Granularity**: Frame-by-frame accuracy analysis
- **Purpose**: Temporal consistency evaluation
- **Output**: Time-series accuracy data

#### 3. Joint-wise PCK
- **Granularity**: Individual joint performance
- **Purpose**: Identify which joints are most/least accurate
- **Output**: Per-joint accuracy statistics

#### 4. Group-wise Analysis
- **Grouping**: By subject, action, and camera
- **Purpose**: Performance analysis across different conditions
- **Output**: Stratified performance metrics

---

## Troubleshooting

### Common Issues

#### 1. **NPZ File Loading Error**
```
KeyError: 'positions_2d'
```

**Solution**: Ensure you downloaded the correct NPZ file from VideoPose3D:
```python
# Verify NPZ structure
import numpy as np
data = np.load("data_2d_humaneva20_gt.npz", allow_pickle=True)
print("Available keys:", list(data.keys()))

# Check data structure
if "positions_2d" in data:
    data_dict = data["positions_2d"].item()
    print("Subjects:", list(data_dict.keys()))
```

#### 2. **Video File Not Found**
```
❌ No GT found for Walking_1_(C1).avi
```

**Solution**: Check video file naming convention and paths:
- Expected format: `{Action}_{Instance}_(C{Camera}).avi`
- Ensure consistent naming across subjects
- Verify file extensions (.avi vs .mp4)

**Debugging:**
```python
import os

def check_video_structure(video_root):
    """Check video file naming patterns"""
    for root, dirs, files in os.walk(video_root):
        for file in files:
            if file.endswith(('.avi', '.mp4')):
                print(f"Found: {file}")
                # Check naming pattern
                if not re.match(r'.+_\(C\d+\)\.avi$', file):
                    print(f"Warning: Non-standard naming: {file}")
```

#### 3. **Sync Data Mismatch**
```
KeyError: No sync index for S1 | Walking 1 | C1
```

**Solution**: Verify sync data configuration in `humaneva_config.yaml`:
```yaml
dataset:
  sync_data:
    data:
      S1:
        "Walking 1": [667, 667, 667]  # Ensure exact action name match
        "Jog 1": [49, 50, 51]
```

**Debugging:**
```python
# Check available actions in ground truth
import pandas as pd
df = pd.read_csv("validate_combined_chunk0.csv")
print("Available action_groups:")
for subject in df['Subject'].unique():
    actions = df[df['Subject'] == subject]['action_group'].unique()
    print(f"  {subject}: {list(actions)}")
```

#### 4. **Empty CSV Output**
**Possible Causes**:
- Incorrect NPZ file format
- Missing chunk0 data
- Path configuration errors

**Debugging:**
```python
# Check NPZ structure
data = np.load(npz_file, allow_pickle=True)
data_dict = data["positions_2d"].item()

print("Subjects:", list(data_dict.keys()))
for subject in data_dict.keys():
    print(f"  {subject} actions:", list(data_dict[subject].keys()))
    
    # Check for chunk0 data
    chunk0_actions = [action for action in data_dict[subject].keys() 
                     if 'chunk0' in action]
    print(f"  {subject} chunk0 actions:", chunk0_actions)
```

#### 5. **Keypoint Array Shape Issues**
```
ValueError: cannot reshape array of size X into shape (Y, Z)
```

**Solution**: Check array dimensions and adapt reshaping:
```python
def safe_reshape_keypoints(array, expected_joints=20):
    """Safely reshape keypoint arrays with dimension checking"""
    if array.ndim == 1:
        # Flat array: reshape to (frames, joints, 2)
        total_coords = len(array)
        frames = total_coords // (expected_joints * 2)
        return array.reshape(frames, expected_joints, 2)
    elif array.ndim == 2:
        # Already 2D: interpret as (frames, coords)
        coords_per_frame = array.shape[1]
        joints = coords_per_frame // 2
        return array.reshape(array.shape[0], joints, 2)
    else:
        return array
```

### Performance Optimization

#### 1. **Large Dataset Processing**
```python
import multiprocessing as mp
from functools import partial

def process_subject_parallel(subjects, process_func):
    """Process multiple subjects in parallel"""
    with mp.Pool(processes=mp.cpu_count() - 1) as pool:
        pool.map(process_func, subjects)
```

#### 2. **Memory Management**
```python
def process_in_chunks(data, chunk_size=1000):
    """Process data in chunks to avoid memory issues"""
    for i in range(0, len(data), chunk_size):
        yield data[i:i+chunk_size]

# Usage
for chunk in process_in_chunks(large_dataset):
    process_chunk(chunk)
```

#### 3. **Caching Strategies**
```python
import pickle
import os

def cached_ground_truth_loader(csv_path, cache_dir="cache"):
    """Cache processed ground truth data for faster loading"""
    cache_file = os.path.join(cache_dir, 
                             f"{os.path.basename(csv_path)}.pkl")
    
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    # Process and cache
    loader = GroundTruthLoader(csv_path)
    os.makedirs(cache_dir, exist_ok=True)
    with open(cache_file, 'wb') as f:
        pickle.dump(loader, f)
    
    return loader
```

---

## Advanced Features

### Custom Action Filtering

Filter specific actions or subjects for targeted evaluation:

```python
def filter_dataset(df, subjects=None, actions=None, cameras=None):
    """Filter dataset by multiple criteria"""
    filtered_df = df.copy()
    
    if subjects:
        filtered_df = filtered_df[filtered_df['Subject'].isin(subjects)]
    if actions:
        filtered_df = filtered_df[filtered_df['action_group'].isin(actions)]
    if cameras:
        filtered_df = filtered_df[filtered_df['Camera'].isin(cameras)]
    
    return filtered_df

# Example usage
walking_only = filter_dataset(df, actions=["Walking 1"])
subject1_only = filter_dataset(df, subjects=["S1"])
```

### Multi-Resolution Support

Handle different video resolutions automatically:

```python
def auto_rescale_keypoints(keypoints, orig_resolution, target_resolution):
    """Automatically rescale keypoints between different resolutions"""
    scale_x = target_resolution[0] / orig_resolution[0]
    scale_y = target_resolution[1] / orig_resolution[1]
    
    keypoints_rescaled = keypoints.copy()
    keypoints_rescaled[:, :, 0] *= scale_x  # X coordinates
    keypoints_rescaled[:, :, 1] *= scale_y  # Y coordinates
    
    return keypoints_rescaled

def batch_rescale_dataset(input_dir, output_dir, target_resolution):
    """Rescale entire dataset to target resolution"""
    for csv_file in glob.glob(os.path.join(input_dir, "*.csv")):
        df = pd.read_csv(csv_file)
        
        # Extract and rescale keypoint columns
        keypoint_cols = [col for col in df.columns if re.match(r'[xy]\d+', col)]
        
        for idx, row in df.iterrows():
            orig_res = get_video_resolution_from_metadata(row)
            if orig_res != target_resolution:
                # Rescale keypoints
                for col in keypoint_cols:
                    if col.startswith('x'):
                        df.loc[idx, col] *= target_resolution[0] / orig_res[0]
                    elif col.startswith('y'):
                        df.loc[idx, col] *= target_resolution[1] / orig_res[1]
        
        # Save rescaled data
        output_path = os.path.join(output_dir, os.path.basename(csv_file))
        df.to_csv(output_path, index=False)
```

### Quality Control Metrics

Implement additional validation checks:

```python
def validate_keypoint_quality(keypoints, video_resolution, 
                            validity_threshold=0.9):
    """Validate keypoint quality and detect outliers"""
    h, w = video_resolution
    
    # Check bounds
    valid_x = np.logical_and(keypoints[:, :, 0] >= 0, 
                           keypoints[:, :, 0] < w)
    valid_y = np.logical_and(keypoints[:, :, 1] >= 0, 
                           keypoints[:, :, 1] < h)
    valid_joints = np.logical_and(valid_x, valid_y)
    
    validity_ratio = np.mean(valid_joints)
    
    # Detect outliers using statistical methods
    coords_x = keypoints[:, :, 0].flatten()
    coords_y = keypoints[:, :, 1].flatten()
    
    # IQR-based outlier detection
    def detect_outliers_iqr(data):
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return np.logical_or(data < lower_bound, data > upper_bound)
    
    outliers_x = detect_outliers_iqr(coords_x[~np.isnan(coords_x)])
    outliers_y = detect_outliers_iqr(coords_y[~np.isnan(coords_y)])
    
    quality_report = {
        'validity_ratio': validity_ratio,
        'outliers_x_ratio': np.mean(outliers_x),
        'outliers_y_ratio': np.mean(outliers_y),
        'passes_quality_check': validity_ratio > validity_threshold
    }
    
    return quality_report

def generate_quality_report(csv_path, video_resolution):
    """Generate comprehensive quality report for dataset"""
    loader = GroundTruthLoader(csv_path)
    df = pd.read_csv(csv_path)
    
    quality_reports = []
    
    for subject in df['Subject'].unique():
        for action_group in df['action_group'].unique():
            for camera in df['Camera'].unique():
                try:
                    keypoints = loader.get_keypoints(
                        subject, action_group, camera, chunk="chunk0"
                    )
                    
                    quality = validate_keypoint_quality(
                        keypoints, video_resolution
                    )
                    
                    quality_reports.append({
                        'subject': subject,
                        'action': action_group,
                        'camera': camera,
                        **quality
                    })
                    
                except Exception as e:
                    print(f"Quality check failed for {subject}, {action_group}, {camera}: {e}")
    
    return pd.DataFrame(quality_reports)
```

### Visualization Enhancements

Create advanced visualizations:

```python
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def create_skeleton_overlay(frame, keypoints, connections):
    """Draw skeletal connections on video frames"""
    # Define bone connections for human skeleton
    bone_pairs = [
        (0, 1),   # Hip connection
        (0, 2),   # Left hip to knee
        (1, 3),   # Right hip to knee
        (2, 4),   # Left knee to ankle
        (3, 5),   # Right knee to ankle
        (8, 9),   # Head to neck
        (9, 10),  # Neck to left shoulder
        (9, 11),  # Neck to right shoulder
        (10, 12), # Left shoulder to elbow
        (11, 13), # Right shoulder to elbow
        (12, 14), # Left elbow to wrist
        (13, 15), # Right elbow to wrist
    ]
    
    # Draw bones
    for start_idx, end_idx in bone_pairs:
        if start_idx < len(keypoints) and end_idx < len(keypoints):
            start_point = tuple(map(int, keypoints[start_idx]))
            end_point = tuple(map(int, keypoints[end_idx]))
            cv2.line(frame, start_point, end_point, (255, 0, 0), 2)
    
    # Draw joints
    for i, (x, y) in enumerate(keypoints):
        cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)
        cv2.putText(frame, str(i), (int(x)+5, int(y)-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    return frame

def create_motion_trajectory_plot(keypoints_sequence, joint_indices=None):
    """Create trajectory plots for specific joints over time"""
    if joint_indices is None:
        joint_indices = [0, 1, 2, 3, 4, 5]  # Hip, knee, ankle joints
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    joint_names = ['Left Hip', 'Right Hip', 'Left Knee', 
                  'Right Knee', 'Left Ankle', 'Right Ankle']
    
    for i, joint_idx in enumerate(joint_indices):
        if i >= len(axes):
            break
            
        x_coords = keypoints_sequence[:, joint_idx, 0]
        y_coords = keypoints_sequence[:, joint_idx, 1]
        
        axes[i].plot(x_coords, y_coords, 'b-', alpha=0.7)
        axes[i].scatter(x_coords[0], y_coords[0], c='green', s=50, label='Start')
        axes[i].scatter(x_coords[-1], y_coords[-1], c='red', s=50, label='End')
        axes[i].set_title(f'{joint_names[i]} Trajectory')
        axes[i].set_xlabel('X coordinate (pixels)')
        axes[i].set_ylabel('Y coordinate (pixels)')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        axes[i].invert_yaxis()  # Invert Y-axis to match image coordinates
    
    plt.tight_layout()
    return fig

def create_animated_skeleton(keypoints_sequence, video_resolution, 
                           output_path="skeleton_animation.gif"):
    """Create animated skeleton visualization"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    def animate(frame_idx):
        ax.clear()
        ax.set_xlim(0, video_resolution[0])
        ax.set_ylim(0, video_resolution[1])
        ax.invert_yaxis()
        
        keypoints = keypoints_sequence[frame_idx]
        
        # Draw skeleton
        bone_pairs = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5)]
        for start_idx, end_idx in bone_pairs:
            start_point = keypoints[start_idx]
            end_point = keypoints[end_idx]
            ax.plot([start_point[0], end_point[0]], 
                   [start_point[1], end_point[1]], 'b-', linewidth=2)
        
        # Draw joints
        ax.scatter(keypoints[:, 0], keypoints[:, 1], c='red', s=50)
        
        ax.set_title(f'Frame {frame_idx}')
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
    
    anim = FuncAnimation(fig, animate, frames=len(keypoints_sequence), 
                        interval=100, repeat=True)
    anim.save(output_path, writer='pillow')
    plt.close()
    
    return output_path
```

---

## Summary

This comprehensive guide provides a complete workflow for setting up HumanEva dataset ground truth data for pose estimation evaluation. The pipeline transforms raw NPZ files into structured, synchronized, and validated ground truth data suitable for robust pose estimation evaluation.

### Key Advantages of This Setup:
- **Standardized Format**: CSV files compatible with pandas and standard data analysis tools
- **Quality Assurance**: Visual overlays and validation metrics for data integrity
- **Synchronization**: Frame-accurate alignment between video and motion capture data
- **Flexibility**: Support for different actions, subjects, and camera views
- **Integration**: Seamless connection with evaluation metrics and machine learning pipelines
- **Extensibility**: Modular design allows for custom processing and analysis

### Workflow Summary:
1. **Data Acquisition**: Download preprocessed NPZ from VideoPose3D
2. **Format Conversion**: NPZ → Individual CSVs → Combined CSV
3. **Quality Control**: Generate visualization overlays
4. **Integration**: Configure pipeline for evaluation
5. **Analysis**: Run comprehensive pose estimation evaluation

### Expected Performance:
- **Processing Time**: ~5-10 minutes for full dataset conversion
- **Storage Requirements**: ~50-100MB for processed CSV files
- **Memory Usage**: ~1-2GB peak during NPZ processing
- **Quality Metrics**: >90% keypoint validity for well-processed data

For additional support, advanced customization, or troubleshooting complex issues, refer to the individual script files and the main evaluation pipeline documentation.
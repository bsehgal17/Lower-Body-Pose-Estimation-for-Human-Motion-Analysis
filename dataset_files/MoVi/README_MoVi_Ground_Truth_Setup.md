# MoVi Dataset Ground Truth Setup Guide

This guide provides comprehensive instructions for setting up ground truth data from the MoVi (Motion and Video) dataset for human pose estimation evaluation. The MoVi dataset contains synchronized 3D motion capture data and multi-view video recordings.

## Table of Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Dataset Structure](#dataset-structure)
- [Step-by-Step Setup](#step-by-step-setup)
- [Understanding Output Files](#understanding-output-files)
- [Joint Mapping](#joint-mapping)
- [Troubleshooting](#troubleshooting)
- [Integration with Pipeline](#integration-with-pipeline)

---

## Overview

The MoVi ground truth setup process involves:
1. **3D-to-2D Projection**: Converting 3D mocap joint positions to 2D image coordinates
2. **Video Synchronization**: Aligning mocap data with video frames using motion segments
3. **Data Export**: Generating `.npy` and `.csv` files compatible with the evaluation pipeline

### Key Features
- Automatic camera calibration using intrinsic and extrinsic parameters
- Motion segmentation (e.g., walking, running) with frame-accurate cropping
- Multiple output formats for compatibility with different evaluation tools
- Visualization overlays for quality verification

---

## Prerequisites

### Required Files
Ensure you have the complete MoVi dataset with the following structure:

```
MoVi dataset/
├── AMASS/                         # 3D joint data in AMASS format
│   ├── F_amass_Subject_001.mat
│   ├── F_amass_Subject_002.mat
│   └── ...
├── Camera Parameters/
│   └── Calib/
│       ├── cameraParams_PG1.npz   # Camera intrinsics
│       └── Extrinsics_PG1.npz     # Camera extrinsics
├── dataverse_files/               # Raw video files
│   ├── F_PG1_Subject_001_L.avi
│   ├── F_PG1_Subject_002_L.avi
│   └── ...
├── MoVi_mocap/                    # Motion segmentation data
│   ├── F_v3d_Subject_001.mat
│   ├── F_v3d_Subject_002.mat
│   └── ...
└── results_all_subjects/         # OUTPUT directory (auto-created)
```

### Dependencies
```bash
pip install opencv-python numpy scipy matplotlib pandas
```

---

## Dataset Structure

### Input File Types

| File Type | Description | Contains |
|-----------|-------------|----------|
| `F_amass_Subject_{ID}.mat` | AMASS motion capture data | 3D joint positions per motion type |
| `cameraParams_PG1.npz` | Camera intrinsics | Intrinsic matrix, distortion coefficients |
| `Extrinsics_PG1.npz` | Camera extrinsics | Rotation matrix, translation vector |
| `F_PG1_Subject_{ID}_L.avi` | Video recordings | Full video sequence for each subject |
| `F_v3d_Subject_{ID}.mat` | Motion segments | Start/end frames for each motion type |

### Camera Parameters
- **Intrinsics** (`cameraParams_PG1.npz`):
  - `IntrinsicMatrix`: 3x3 camera matrix
  - `RadialDistortion`: Radial distortion coefficients
  - `TangentialDistortion`: Tangential distortion coefficients

- **Extrinsics** (`Extrinsics_PG1.npz`):
  - `rotationMatrix`: 3x3 rotation matrix (world → camera)
  - `translationVector`: 3x1 translation vector

---

## Step-by-Step Setup

### Step 1: Configure Dataset Paths

Edit the paths in `MoVi_setup_all.py`:

```python
if __name__ == "__main__":
    base = r"C:\path\to\your\MoVi dataset"  # Update this path
    process_all_subjects(
        intr_dir=os.path.join(base, "Camera Parameters", "Calib"),
        extr_dir=os.path.join(base, "Camera Parameters", "Calib"),
        amass_dir=os.path.join(base, "AMASS"),
        video_dir=os.path.join(base, "dataverse_files"),
        v3d_dir=os.path.join(base, "MoVi_mocap"),
        output_root=os.path.join(base, "results_all_subjects"),
        motion="walking",  # Change motion type if needed
    )
```

### Step 2: Run Ground Truth Generation

Execute the setup script:

```bash
python MoVi_setup_all.py
```

The script will:
1. Load camera calibration parameters
2. Extract 3D joint data from AMASS files
3. Parse motion segments (start/end frames)
4. Crop videos to specific motion segments
5. Project 3D joints to 2D image coordinates
6. Generate output files for each subject

### Step 3: Verify Output Structure

After successful execution, you should see:

```
results_all_subjects/
├── Subject_001/
│   ├── walking_cropped.avi        # Cropped video segment
│   ├── joints2d_projected.npy     # 2D joints (NumPy format)
│   ├── joints2d_projected.csv     # 2D joints (CSV format)
│   └── walking_overlay.avi        # Video with joint overlays
├── Subject_002/
│   └── ...
└── ...
```

### Step 4: Convert NPY to CSV (Optional)

If you need additional CSV processing:

```python
import numpy as np
import pandas as pd

def convert_npy_to_csv(npy_path):
    """Convert .npy joint data to structured CSV format"""
    data = np.load(npy_path)  # Shape: (frames, joints, 2)
    flattened = data.reshape((data.shape[0], -1))
    
    # Generate column headers
    headers = []
    for j in range(data.shape[1]):
        headers += [f"joint{j}_x", f"joint{j}_y"]
    
    df = pd.DataFrame(flattened, columns=headers)
    csv_path = npy_path.replace(".npy", ".csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")

# Usage
convert_npy_to_csv("results_all_subjects/Subject_001/joints2d_projected.npy")
```

---

## Understanding Output Files

### File Formats

#### 1. `joints2d_projected.npy`
- **Format**: NumPy array
- **Shape**: `(frames, joints, 2)`
- **Content**: 2D joint coordinates (x, y) in image space
- **Usage**: Direct loading in Python for evaluation

```python
import numpy as np
joints = np.load("joints2d_projected.npy")
print(f"Shape: {joints.shape}")  # (frames, 22, 2) for MoVi
```

#### 2. `joints2d_projected.csv`
- **Format**: CSV with flat structure
- **Columns**: `joint0_x, joint0_y, joint1_x, joint1_y, ...`
- **Rows**: Each row represents one frame
- **Usage**: Compatible with pandas and spreadsheet applications

#### 3. `walking_cropped.avi`
- **Format**: Video file (XVID codec)
- **Content**: Video segment corresponding to the specific motion
- **Frame Rate**: Preserved from original video
- **Usage**: Visual verification and analysis

#### 4. `walking_overlay.avi`
- **Format**: Video file with joint overlays
- **Content**: Original video with 2D joints rendered as green circles
- **Usage**: Quality control and visualization

---

## Joint Mapping

### MoVi Joint Layout (GTJointsMoVi)

The MoVi dataset uses a specific joint indexing system:

```python
class GTJointsMoVi(Enum):
    HEAD = 15
    NECK = 14
    LEFT_SHOULDER = 16
    RIGHT_SHOULDER = 17
    LEFT_ELBOW = 18
    RIGHT_ELBOW = 19
    LEFT_WRIST = 20
    RIGHT_WRIST = 21
    LEFT_HIP = 1
    RIGHT_HIP = 2
    LEFT_KNEE = 4
    RIGHT_KNEE = 5
    LEFT_ANKLE = 7
    RIGHT_ANKLE = 8
    LEFT_TOE = 10    # 2nd metatarsal
    RIGHT_TOE = 11
```

### Important Notes
- Joint indices are **0-based** in the exported arrays
- The original AMASS data contains ~22 joints per frame
- Lower body joints (hips, knees, ankles) are primary focus for this pipeline
- Coordinate system: (0,0) is top-left corner of image

---

## Troubleshooting

### Common Issues

#### 1. **Motion Not Found Error**
```
ValueError: Motion 'walking' not found in Subject_001
```

**Solution**: Check available motions in the AMASS file:
```python
mat = scipy.io.loadmat("F_amass_Subject_001.mat", struct_as_record=False, squeeze_me=True)
subject = mat["Subject_001"]
motions = [m.description for m in subject.move if hasattr(m, 'description')]
print("Available motions:", motions)
```

#### 2. **Video File Not Found**
```
❌ Skipping Subject 001: video not found.
```

**Solution**: Verify video file naming convention and paths:
- Expected format: `F_PG1_Subject_{ID}_L.avi`
- Check for consistent zero-padding in subject IDs

#### 3. **Camera Parameter Loading Error**
```
FileNotFoundError: cameraParams_PG1.npz not found
```

**Solution**: Ensure camera calibration files are in the correct subdirectory:
```
Camera Parameters/
└── Calib/
    ├── cameraParams_PG1.npz
    └── Extrinsics_PG1.npz
```

#### 4. **Empty Output Arrays**
**Possible Causes**:
- Incorrect motion segmentation timing
- Camera projection matrix issues
- Mocap-video synchronization problems

**Debugging**:
```python
# Check motion segments
segments = parse_v3d_segments("F_v3d_Subject_001.mat")
print("Motion segments:", segments)

# Verify joint data shape
joints3d, _ = extract_amass_joints("F_amass_Subject_001.mat", "walking")
print("3D joints shape:", joints3d.shape)
```

### Performance Optimization

#### 1. **Processing Large Datasets**
- Process subjects in batches to manage memory usage
- Use multiprocessing for parallel subject processing
- Consider cropping video segments first to reduce file I/O

#### 2. **Storage Considerations**
- NPY files: ~1-5MB per subject (depending on sequence length)
- Video files: ~10-50MB per cropped segment
- Plan for ~100-500MB total storage per subject

---

## Integration with Pipeline

### Configuration Setup

Update your pipeline configuration (`MoVi_test.yaml`):

```yaml
paths:
  dataset: MoVi
  ground_truth_file: /path/to/MoVi/results_all_subjects/

dataset:
  joint_enum_module: utils.joint_enum.GTJointsMoVi
  keypoint_format: utils.joint_enum.PredJointsDeepLabCut

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

### Loading Ground Truth in Evaluation

The ground truth data is automatically loaded by the evaluation pipeline:

```python
from dataset_files.MoVi.movi_assessor import assess_single_movi_sample

# Example usage
gt_keypoints, _, _, pred_keypoints, _, _ = assess_single_movi_sample(
    gt_csv_path="results_all_subjects/Subject_001/joints2d_projected.csv",
    pred_pkl_path="predictions/Subject_001_walking.pkl",
    video_path="videos/Subject_001_walking.avi"
)
```

### Evaluation Metrics

The pipeline supports multiple evaluation metrics for MoVi data:
- **PCK (Percentage of Correct Keypoints)**: Multiple threshold values
- **Per-frame PCK**: Frame-by-frame accuracy analysis
- **Joint-wise metrics**: Individual joint performance
- **Overall mAP**: Mean Average Precision across all joints

### Expected Performance

Typical ground truth generation times:
- **Per subject**: 30-60 seconds
- **Full dataset (~80 subjects)**: 30-60 minutes
- **Bottlenecks**: Video I/O and 3D-to-2D projection calculations

---

## Advanced Usage

### Custom Motion Types

To process different motion types:

```python
# Available motion types in MoVi
motion_types = ["walking", "running", "jumping", "sitting", "standing"]

for motion in motion_types:
    process_all_subjects(
        # ... same parameters ...
        motion=motion
    )
```

### Multiple Camera Views

For multi-view setups, modify the camera parameter loading:

```python
# Process different camera positions
cameras = ["PG1", "PG2", "PG3"]  # Example camera IDs

for cam in cameras:
    intr_path = f"cameraParams_{cam}.npz"
    extr_path = f"Extrinsics_{cam}.npz"
    # ... continue processing ...
```

### Quality Control

Implement additional quality checks:

```python
def validate_projection_quality(joints2d, video_frames):
    """Validate that projected joints are within image boundaries"""
    h, w = video_frames[0].shape[:2]
    
    # Check bounds
    valid_x = np.logical_and(joints2d[:, :, 0] >= 0, joints2d[:, :, 0] < w)
    valid_y = np.logical_and(joints2d[:, :, 1] >= 0, joints2d[:, :, 1] < h)
    valid_joints = np.logical_and(valid_x, valid_y)
    
    validity_ratio = np.mean(valid_joints)
    print(f"Joint validity ratio: {validity_ratio:.3f}")
    
    return validity_ratio > 0.8  # 80% threshold
```

---

## Summary

This guide covers the complete process of setting up MoVi dataset ground truth data for pose estimation evaluation. The generated files provide accurate 2D joint annotations synchronized with video data, enabling robust evaluation of pose estimation algorithms on realistic human motion sequences.

For additional support or advanced customization, refer to the source code in `MoVi_setup_all.py` and the evaluation pipeline documentation.

# HumanSC3D Dataset Handler

This directory contains the implementation for handling the HumanSC3D dataset in the Lower-Body Pose Estimation pipeline.

## Dataset Structure

The HumanSC3D dataset has the following structure:
```
humansc3d/
├── humansc3d_train/
│   └── train/
│       ├── s01/
│       │   ├── videos/
│       │   │   ├── 50591643/
│       │   │   │   ├── 170.mp4
│       │   │   │   ├── 001.mp4
│       │   │   │   └── ...
│       │   │   ├── 58860488/
│       │   │   └── ...
│       │   └── processed_outputs/
│       │       └── 2d_points/
│       │           ├── 170_50591643_2d.json
│       │           ├── 001_50591643_2d.json
│       │           └── ...
│       ├── s02/
│       └── ...
```

## Files Description

### `humansc3d_metadata.py`
Contains functions to parse video file paths and extract metadata:
- `get_humansc3d_metadata_from_video()`: Extracts subject, action, and camera from video path
- `get_humansc3d_gt_path()`: Generates corresponding ground truth JSON path

### `humansc3d_gt_loader.py`
Handles loading of ground truth keypoints from JSON files:
- `HumanSC3DGroundTruthLoader`: Main class for loading GT data
- `load_humansc3d_gt_keypoints()`: Convenience function

### `humansc3d_evaluation.py`
Evaluation pipeline specific to HumanSC3D:
- `humansc3d_data_loader()`: Loads and prepares GT and prediction data
- `run_humansc3d_assessment()`: Main evaluation function

### Configuration Files
- `humansc3d_config.yaml`: Configuration for RTMW detector
- `humansc3d_config_mediapipe.yaml`: Configuration for MediaPipe detector

## Ground Truth Format

The ground truth files are JSON files with the following structure:
```json
{
  "2d_keypoints": [
    [
      [x1, y1], [x2, y2], ..., [x25, y25]  // Frame 1
    ],
    [
      [x1, y1], [x2, y2], ..., [x25, y25]  // Frame 2
    ],
    ...
  ]
}
```

## Joint Format

HumanSC3D uses a 25-joint format as defined in `utils.joint_enum.GTJointsHumanSC3D`:
- 0: PELVIS
- 1: LEFT_HIP
- 2: LEFT_KNEE
- 3: LEFT_ANKLE
- 4: RIGHT_HIP
- 5: RIGHT_KNEE
- 6: RIGHT_ANKLE
- 7: HEAD
- 8: NECK
- 9: SPINE
- 10: SPINE_TOP
- 11: LEFT_SHOULDER
- 12: LEFT_ELBOW
- 13: LEFT_WRIST
- 14: RIGHT_SHOULDER
- 15: RIGHT_ELBOW
- 16: RIGHT_WRIST
- 17-24: Hand and foot extremities

## Usage

To run evaluation on HumanSC3D dataset:

1. Ensure your data follows the expected directory structure
2. Update the `ground_truth_file` path in the YAML configuration
3. Run the pipeline with the appropriate config file:
   ```bash
   python main.py --config dataset_files/HumanSC3D/humansc3d_config.yaml
   ```

## Example Paths

- Video: `C:/Users/BhavyaSehgal/Downloads/humansc3d/humansc3d_train/train/s06/videos/50591643/170.mp4`
- Ground Truth: `C:/Users/BhavyaSehgal/Downloads/humansc3d/humansc3d_train/train/s06/processed_outputs/2d_points/170_50591643_2d.json`

The system automatically handles the mapping between video files and their corresponding ground truth files.
# 🕺 2D Joint Projection Pipeline from MoVi Dataset

The script MoVi_setup_all.py  extracts and projects 3D motion capture (MoCap) joints onto 2D video frames using the **MoVi Dataset**, camera calibration files, and motion segmentation data.

---

## 📌 Overview

### 🔧 What It Does

* Extracts **3D joint positions** from AMASS-formatted `.mat` files.
* Projects 3D joints to 2D image coordinates using camera **intrinsics** and **extrinsics**.
* Crops the relevant segment of the **video** (e.g., walking only).
* Overlays the 2D joints on video frames.
* Saves the final joint coordinates as `.npy` and `.csv`.

---

## 🗂 Directory Structure

```
MoVi dataset/
├── AMASS/                     # Contains AMASS-style .mat joint files
├── Camera Parameters/
│   └── Calib/
│       ├── cameraParams_PG1.npz
│       └── Extrinsics_PG1.npz
├── dataverse_files/           # Raw videos (.avi)
├── MoVi_mocap/                # V3D motion segment files (.mat)
└── results_all_subjects/     # OUTPUT: joints, overlay, cropped video
```

---

## 📥 Input Files

| File                       | Description                                         |
| -------------------------- | --------------------------------------------------- |
| `cameraParams_PG1.npz`     | Contains intrinsic matrix and distortion parameters |
| `Extrinsics_PG1.npz`       | Contains rotation and translation matrix            |
| `F_amass_Subject_{ID}.mat` | 3D joint positions per motion (AMASS format)        |
| `F_PG1_Subject_{ID}_L.avi` | Full video of subject                               |
| `F_v3d_Subject_{ID}.mat`   | Contains start and end frames for each motion       |

---

## 🧠 How It Works

### 1. Load Camera Parameters

```python
K, dist = load_camera_matrix_npz("cameraParams_PG1.npz")
R, T = load_extrinsics_npz("Extrinsics_PG1.npz")
```

### 2. Load 3D Joints

```python
joints3d, _ = extract_amass_joints("F_amass_Subject_ID.mat", motion="walking")
```

### 3. Parse Motion Segment

```python
segments = parse_v3d_segments("F_v3d_Subject_ID.mat")
start_frame, end_frame = segments["walking"]
```

### 4. Crop the Video

```python
extract_video_segment(video_path, output_path, start_sec, duration_sec)
```

### 5. Project 3D → 2D

```python
joints2d = project_3d_to_2d(joints3d, K, R, T, fps=30, mocap_fps=120)
```

### 6. Overlay and Save

```python
overlay_video_with_joints(frames, joints2d_trimmed, overlay_path)
np.save("joints2d_projected.npy", joints2d_trimmed)
```

---

## 📤 Convert `.npy` to `.csv`

### `convert_npy_to_csv.py`

```python
import numpy as np
import pandas as pd
import os

def convert_npy_to_csv(npy_path):
    data = np.load(npy_path)  # Shape: (frames, joints, 2)
    flattened = data.reshape((data.shape[0], -1))

    headers = []
    for j in range(data.shape[1]):
        headers += [f"joint{j}_x", f"joint{j}_y"]

    df = pd.DataFrame(flattened, columns=headers)
    csv_path = npy_path.replace(".npy", ".csv")
    df.to_csv(csv_path, index=False)
    print(f"✅ Saved CSV: {csv_path}")

# Example
convert_npy_to_csv("results_all_subjects/Subject_001/joints2d_projected.npy")
```

---

## ✅ Outputs per Subject

| Output File              | Description                                |
| ------------------------ | ------------------------------------------ |
| `walking_cropped.avi`    | Cropped segment of the walking motion      |
| `joints2d_projected.npy` | 2D joint projections (NumPy format)        |
| `joints2d_projected.csv` | Same joint data in CSV format              |
| `walking_overlay.avi`    | Visualization: 2D joints overlaid on video |

---

## 🚀 Run the Script

### Main Entry Point

```bash
python main.py
```

```python
# main.py
if __name__ == "__main__":
    base = r"C:\path\to\MoVi dataset"
    process_all_subjects(
        intr_dir=os.path.join(base, "Camera Parameters", "Calib"),
        extr_dir=os.path.join(base, "Camera Parameters", "Calib"),
        amass_dir=os.path.join(base, "AMASS"),
        video_dir=os.path.join(base, "dataverse_files"),
        v3d_dir=os.path.join(base, "MoVi_mocap"),
        output_root=os.path.join(base, "results_all_subjects"),
        motion="walking",
    )
```


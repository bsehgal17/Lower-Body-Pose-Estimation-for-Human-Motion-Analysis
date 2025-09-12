--# HumanEva Dataset Ground Truth Preparation

This document explains how we obtained and processed the **HumanEva** dataset ground truth for use in our lower-body pose estimation pipeline. The data is downloaded as NPZ files from video3d, then NPY files are extracted, converted to structured `.csv` files using npy_to_csv script, and finally filtered and combined into a single reference file for testing using get_combined_csv.

---

## Step 1: Download NPZ File from Video3D

Download the NPZ file from [VideoPose3D](https://github.com/facebookresearch/VideoPose3D) following their official guide from [VideoPose3D's DATASETS.md](https://github.com/facebookresearch/VideoPose3D/blob/main/DATASETS.md) to obtain the `data_3d_humaneva.npz` file, which contains structured 3D keypoint annotations for HumanEva.manEva Dataset Ground Truth Preparation

This document explains how we obtained and processed the **HumanEva** dataset ground truth for use in our lower-body pose estimation pipeline. The data is extracted from preprocessed `.npz` files originally prepared using instructions from [VideoPose3D](https://github.com/facebookresearch/VideoPose3D), then converted to structured `.csv` files, and finally filtered and combined into a single reference file for evaluation.

---

## Step 1: Obtain Ground Truth (NPZ Format)

We followed the official guide from [VideoPose3D’s DATASETS.md](https://github.com/facebookresearch/VideoPose3D/blob/main/DATASETS.md) to generate the `data_3d_humaneva.npz` file, which contains structured 3D keypoint annotations for HumanEva.

This file includes:

* All 3D joint positions
* Grouped by subject (S1, S2, S3), action (Walking, Jog), and camera index (C1, C2, C3)
* Motion capture data aligned with video sequences

---

## Step 2: Extract NPY File

Extract the NPY file from the downloaded NPZ file to access the structured keypoint data.

---

## Step 3: Convert NPY to CSV Using npy_to_csv Script

Use the **npy_to_csv script** to convert the extracted NPY data into individual `.csv` files. This script reads the hierarchical structure of the NPY file and writes frame-wise 2D and 3D keypoints to CSV.

For each `(subject, action, camera)` triplet, we generate a dedicated CSV file with:

* 2D keypoints (`x1, y1, x2, y2, ..., xN, yN`)
* Metadata columns like `Subject`, `Action`, `Camera`, and `action_group`

This format makes it easy to inspect and process the motion capture data using common tools like pandas or spreadsheets.

---

## Step 4: Get Combined CSV for Ground Truth Testing

The HumanEva dataset provides action sequences divided into chunks (e.g., `chunk0`, `chunk1`, etc.). As recommended by the original authors and consistent with VideoPose3D evaluation, we use **only `chunk0`**, which corresponds to the **most validated and reliable motion capture segment**.

To combine all relevant CSVs and get the combined ground truth file for testing, we use this script:

📎 [`get_combined_csv.py`](https://github.com/bsehgal17/Lower-Body-Pose-Estimation-for-Human-Motion-Analysis/blob/main/dataset_files/HumanEva/get_combined_csv.py)

This script:

* Scans through all generated per-action CSVs
* Filters rows with `"chunk0"` in the `Action` column
* Concatenates them into a single unified CSV file for ground truth testing:

  ```
  humaneva_combined_chunk0.csv
  ```

---

## Output

* `humaneva_combined_chunk0.csv`
  → This is the final, readable, and structured ground truth file used for testing.

It contains motion capture keypoints for all subjects, actions, and cameras, but only the `chunk0` portions for consistency and testing integrity.

---

## Usage in Pipeline

This combined CSV is loaded by our keypoint loader during evaluation:

```python
from dataset_files.HumanEva.get_gt_keypoint import GroundTruthLoader

loader = GroundTruthLoader("path/to/humaneva_combined_chunk0.csv")
keypoints = loader.get_keypoints(subject="S1", action="Walking 1", camera_idx=0, chunk="chunk0")
```

This provides an `(N, J, 2)` or `(N, J, 3)` array of 2D/3D keypoints for a given clip, aligned with our prediction pipeline.

---

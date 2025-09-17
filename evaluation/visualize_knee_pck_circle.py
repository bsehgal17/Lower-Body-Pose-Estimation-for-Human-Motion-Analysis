from dataset_files.HumanEva.get_gt_keypoint import GroundTruthLoader
from config.global_config import get_global_config
from utils.joint_enum import GTJointsHumanEVa
from glob import glob
import yaml
import numpy as np
import cv2
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

"""
Script to visualize the norm length and draw a circle around the knee keypoint
with ground truth as center and radius as pck_threshold * norm_length for the first frame of each video.
"""

# Load config from YAML
CONFIG_PATH = "dataset_files/HumanEva/humaneva_config.yaml"
GLOBAL_CONFIG_PATH = "config_yamls/global_config.yaml"
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# Use global config for video dir
global_config = get_global_config(GLOBAL_CONFIG_PATH)
VIDEO_DIR = global_config.paths.input_dir
GT_PATH = config["paths"].get("ground_truth_file", None)
# Add this to your YAML if missing
PCK_THRESHOLD = config.get("pck_threshold", 0.2)
OUTPUT_DIR = "test_plots_output/pck_circles"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_ground_truth_keypoints(video_path, gt_path):
    # Extract subject, action, camera from video filename (assumes format S1_Jog_1_(C1).mp4)
    basename = os.path.basename(video_path)
    parts = basename.split("_")
    subject = parts[0]
    action = f"{parts[1]} {parts[2]}"  # e.g., Jog 1
    camera = int(parts[3].replace("(C", "").replace(")", ""))

    loader = GroundTruthLoader(gt_path)
    keypoints = loader.get_keypoints(subject, action, camera, chunk="chunk0")
    # Use first frame
    frame_kpts = keypoints[0]  # shape (J, 2)
    # Get knee and hip joint indices from enum
    left_knee_idx = GTJointsHumanEVa.LEFT_KNEE.value
    left_hip_idx = GTJointsHumanEVa.LEFT_HIP.value

    # If enum value is tuple, take mean of both points
    def get_joint_xy(idx):
        if isinstance(idx, tuple):
            pts = [frame_kpts[i] for i in idx]
            return np.mean(pts, axis=0)
        else:
            return frame_kpts[idx]

    knee_xy = get_joint_xy(left_knee_idx)
    hip_xy = get_joint_xy(left_hip_idx)
    return {"knee": knee_xy, "hip": hip_xy}

    # ...existing code...


def process_videos(video_dir, output_dir, pck_threshold):
    video_files = glob(os.path.join(video_dir, "*.mp4"))
    for video_path in video_files:
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to read first frame: {video_path}")
            continue
        gt_kpts = get_ground_truth_keypoints(video_path, GT_PATH)
        norm_length = compute_norm_length(gt_kpts)
        vis_img = visualize_frame(frame, gt_kpts, norm_length, pck_threshold)
        out_path = os.path.join(
            output_dir, os.path.basename(video_path) + "_pck_circle.png"
        )
        cv2.imwrite(out_path, vis_img)
        print(f"Saved visualization: {out_path}")
        cap.release()


if __name__ == "__main__":
    process_videos(VIDEO_DIR, OUTPUT_DIR, PCK_THRESHOLD)


# Remove duplicate config and imports


def compute_norm_length(gt_kpts):
    # Example: norm length is hip-to-knee distance
    knee = np.array(gt_kpts["knee"])
    hip = np.array(gt_kpts["hip"])
    return np.linalg.norm(knee - hip)


def visualize_frame(frame, gt_kpts, norm_length, pck_threshold):
    img = frame.copy()
    knee = tuple(map(int, gt_kpts["knee"]))
    # Draw norm line (hip to knee)
    hip = tuple(map(int, gt_kpts["hip"]))
    cv2.line(img, hip, knee, (0, 255, 0), 2)
    # Draw circle around knee
    radius = int(pck_threshold * norm_length)
    cv2.circle(img, knee, radius, (0, 0, 255), 2)
    # Draw keypoints
    cv2.circle(img, knee, 5, (255, 0, 0), -1)
    cv2.circle(img, hip, 5, (255, 255, 0), -1)
    return img


def process_videos(video_dir, output_dir, pck_threshold):
    video_files = glob(os.path.join(video_dir, "*.mp4"))
    for video_path in video_files:
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to read first frame: {video_path}")
            continue
        gt_kpts = get_ground_truth_keypoints(video_path, GT_PATH)
        norm_length = compute_norm_length(gt_kpts)
        vis_img = visualize_frame(frame, gt_kpts, norm_length, pck_threshold)
        out_path = os.path.join(
            output_dir, os.path.basename(video_path) + "_pck_circle.png"
        )
        cv2.imwrite(out_path, vis_img)
        print(f"Saved visualization: {out_path}")
        cap.release()


if __name__ == "__main__":
    process_videos(VIDEO_DIR, OUTPUT_DIR, PCK_THRESHOLD)

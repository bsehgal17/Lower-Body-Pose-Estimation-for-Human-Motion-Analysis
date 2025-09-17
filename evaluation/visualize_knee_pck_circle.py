import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # noqa

from glob import glob
import yaml
import numpy as np
import cv2

from utils.joint_enum import GTJointsHumanEVa
from config.global_config import get_global_config
from dataset_files.HumanEva.get_gt_keypoint import GroundTruthLoader

"""
Script to visualize the norm length and draw a circle around the knee keypoint
with ground truth as center and radius as pck_threshold * norm_length for the first frame of each video.
"""

# ---------------------- Configuration ---------------------- #
CONFIG_PATH = "dataset_files/HumanEva/humaneva_config.yaml"
GLOBAL_CONFIG_PATH = "config_yamls/global_config.yaml"

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)
dataset_name = config["paths"]["dataset"]
global_config = get_global_config(GLOBAL_CONFIG_PATH)
VIDEO_DIR = os.path.join(global_config.paths.input_dir, dataset_name)
GT_PATH = config["paths"].get("ground_truth_file", None)
SYNC_DATA = config.get("dataset", {}).get(
    "sync_data", {})  # <-- sync data dictionary
PCK_THRESHOLD = config.get("pck_threshold", 0.25)
OUTPUT_DIR = "/storage/Projects/Gaitly/bsehgal/lower_body_pose_est/pipeline_results/HumanEva/test_plots_output/pck_circles"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------- Helper Functions ---------------------- #
def get_ground_truth_keypoints(video_path, gt_path):
    """
    Extracts ground truth keypoints for the first frame of the video.
    Returns None if no keypoints found.
    """
    try:
        subject = os.path.basename(
            os.path.dirname(os.path.dirname(video_path)))
        basename = os.path.basename(video_path)  # e.g. Gestures_1_(C3).avi
        name, _ = os.path.splitext(basename)
        parts = name.split("_")

        if len(parts) < 3:
            print(f"[Warning] Unexpected filename format: {basename}")
            return None

        action = f"{parts[0]} {parts[1]}"  # "Gestures 1"
        try:
            camera = int(parts[2].replace("(C", "").replace(")", ""))
        except ValueError:
            print(f"[Warning] Could not parse camera from {basename}")
            return None

        loader = GroundTruthLoader(gt_path)
        keypoints = loader.get_keypoints(
            subject, action, camera, chunk="chunk0")

        if keypoints is None or len(keypoints) == 0:
            print(f"[Warning] No GT keypoints found for {video_path}")
            return None

        frame_kpts = keypoints[0]  # shape (J, 2)

        left_knee_idx = GTJointsHumanEVa.LEFT_KNEE.value
        left_hip_idx = GTJointsHumanEVa.LEFT_HIP.value

        def get_joint_xy(idx):
            if isinstance(idx, tuple):
                pts = [frame_kpts[i] for i in idx if i < len(frame_kpts)]
                return np.mean(pts, axis=0) if pts else None
            else:
                if idx >= len(frame_kpts):
                    return None
                return frame_kpts[idx]

        knee = get_joint_xy(left_knee_idx)
        hip = get_joint_xy(left_hip_idx)

        if knee is None or hip is None:
            print(f"[Warning] Missing joints in GT for {video_path}")
            return None

        return {"knee": knee, "hip": hip}

    except Exception as e:
        print(f"[Error] Failed to load GT for {video_path}: {e}")
        return None


def compute_norm_length(gt_kpts):
    """
    Computes normalization length as the Euclidean distance between hip and knee.
    Returns None if inputs are invalid.
    """
    if gt_kpts is None:
        return None
    try:
        knee = np.array(gt_kpts["knee"])
        hip = np.array(gt_kpts["hip"])
        return np.linalg.norm(knee - hip)
    except Exception:
        return None


def visualize_frame(frame, gt_kpts, norm_length, pck_threshold):
    """
    Draws the norm line and circle for the knee keypoint.
    Returns None if inputs are invalid.
    """
    if gt_kpts is None or norm_length is None:
        return None

    img = frame.copy()
    knee = tuple(map(int, gt_kpts["knee"]))
    hip = tuple(map(int, gt_kpts["hip"]))

    # Draw norm line (hip to knee)
    cv2.line(img, hip, knee, (0, 255, 0), 2)
    # Draw circle around knee based on PCK threshold
    radius = int(pck_threshold * norm_length)
    cv2.circle(img, knee, radius, (0, 0, 255), 2)
    # Draw keypoints
    cv2.circle(img, knee, 5, (255, 0, 0), -1)
    cv2.circle(img, hip, 5, (255, 255, 0), -1)

    return img


def get_sync_frame(subject, action, camera):
    """
    Get sync frame index for given subject, action, and camera from SYNC_DATA.
    Returns None if not available.
    """
    if subject not in SYNC_DATA["data"]:
        return None
    subject_data = SYNC_DATA["data"][subject]
    if action not in subject_data:
        return None
    frames = subject_data[action]
    if camera - 1 < len(frames):
        return frames[camera - 1]  # cameras are 1-based
    return None


def process_videos(video_dir, output_dir, pck_threshold):
    """
    Processes all videos in the directory and saves visualizations
    for the sync frame of each video (if available).
    Saves each result inside a subject-specific subfolder.
    """
    video_files = glob(os.path.join(video_dir, "**", "*.avi"), recursive=True)
    for video_path in video_files:
        # Subject (S1, S2, etc.)
        subject = os.path.basename(
            os.path.dirname(os.path.dirname(video_path)))
        basename = os.path.basename(video_path)  # e.g., Gestures_1_(C3).avi
        name, _ = os.path.splitext(basename)
        parts = name.split("_")
        action = f"{parts[0]} {parts[1]}"   # e.g., "Walking 1"
        camera = int(parts[2].replace("(C", "").replace(")", ""))  # e.g., 3

        # Get sync frame index
        sync_frame_idx = get_sync_frame(subject, action, camera)
        if sync_frame_idx is None:
            print(f"[Skip] No sync frame for {video_path}")
            continue

        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, sync_frame_idx)  # jump to sync frame
        ret, frame = cap.read()
        if not ret:
            print(
                f"[Error] Could not read sync frame {sync_frame_idx} in {video_path}")
            cap.release()
            continue

        gt_kpts = get_ground_truth_keypoints(video_path, GT_PATH)
        if gt_kpts is None:
            print(f"[Skip] No GT for {video_path}")
            cap.release()
            continue

        norm_length = compute_norm_length(gt_kpts)
        if norm_length is None or norm_length == 0:
            print(f"[Skip] Invalid norm length for {video_path}")
            cap.release()
            continue

        vis_img = visualize_frame(frame, gt_kpts, norm_length, pck_threshold)
        if vis_img is None:
            print(f"[Skip] Visualization failed for {video_path}")
            cap.release()
            continue

        # Save inside subject-specific folder
        subject_dir = os.path.join(output_dir, subject)
        os.makedirs(subject_dir, exist_ok=True)

        out_path = os.path.join(
            subject_dir, os.path.basename(
                video_path) + f"_sync{sync_frame_idx}_pck_circle.png"
        )
        cv2.imwrite(out_path, vis_img)
        print(f"Saved visualization: {out_path}")

        cap.release()


# ---------------------- Main Execution ---------------------- #
if __name__ == "__main__":
    process_videos(VIDEO_DIR, OUTPUT_DIR, PCK_THRESHOLD)

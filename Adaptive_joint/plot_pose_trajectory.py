import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- USER SETTINGS ---
json_files = [
    # RAW
    "/storageh100/Projects/Gaitly/bsehgal/pipeline_results/HumanEva/detect_RTMW/detect/2025-09-18_00-46-59/S2/Image_Data/Box_1_(C1)/Box_1_(C1).json",
    # UNIFORM
    "/storageh100/Projects/Gaitly/bsehgal/pipeline_results/HumanEva/Butterworth_filter/filter_butterworth_18th_9hz/filter/2026-01-02_16-58-47/butterworth_order18_cutoff7.0_fs60.0/S2/Image_Data/Box_1_(C1)/Box_1_(C1).json",
    # ADAPTIVE
    "/storage/Projects/Gaitly/bsehgal/lower_body_pose_est/pipeline_results/Adaptive_filt/adaptive_filtering/S2/Image_Data/Box_1_(C1)/Box_1_(C1).json"
]

labels = ["Raw", "Uniform Butterworth", "Adaptive"]
colors = ["tab:blue", "tab:orange", "tab:green"]

joint_index = 11
output_svg = "joint11_x_comparison.svg"
# ---------------------


def load_pose(json_path):
    with open(json_path, "r") as f:
        return json.load(f)


def extract_joint_xy_trajectory(pose_data, joint_index):
    if "persons" not in pose_data or not pose_data["persons"]:
        return np.empty((0, 2))

    primary_person = max(pose_data["persons"],
                         key=lambda p: len(p.get("poses", [])))
    poses = sorted(primary_person.get("poses", []),
                   key=lambda p: p["frame_idx"])

    trajectory = []

    for pose in poses:
        kps = pose.get("keypoints", [])
        if isinstance(kps, list) and len(kps) == 1:
            kps = kps[0]

        if len(kps) <= joint_index:
            continue

        kp = kps[joint_index]
        if len(kp) < 2:
            continue

        x, y = float(kp[0]), float(kp[1])
        if not (np.isnan(x) or np.isnan(y)):
            trajectory.append([x, y])

    return np.asarray(trajectory, dtype=np.float32)


def main():

    # Load trajectories
    trajectories = []
    for json_file in json_files:
        pose = load_pose(json_file)
        traj = extract_joint_xy_trajectory(pose, joint_index)
        trajectories.append(traj)

    # Create figure
    fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True)

    # --- Individual stacked plots ---
    for idx, (traj, label, color) in enumerate(zip(trajectories, labels, colors)):

        if traj.shape[0] == 0:
            print(f"No data for {label}")
            continue

        x_coords = traj[:, 0]
        frames = np.arange(len(x_coords))

        axes[idx].plot(frames, x_coords, color=color, linewidth=1.6)
        axes[idx].set_ylabel("X position")
        axes[idx].set_title(label)
        axes[idx].grid(True)

    # --- Overlap plot ---
    ax_overlap = axes[3]

    for traj, label, color in zip(trajectories, labels, colors):

        if traj.shape[0] == 0:
            continue

        x_coords = traj[:, 0]
        frames = np.arange(len(x_coords))

        ax_overlap.plot(frames, x_coords, color=color,
                        linewidth=1.6, label=label)

    ax_overlap.set_title("All Methods Overlaid")
    ax_overlap.set_ylabel("X position")
    ax_overlap.set_xlabel("Frame Index")
    ax_overlap.legend()
    ax_overlap.grid(True)

    fig.suptitle(f"Joint {joint_index} — X Coordinate Comparison", fontsize=14)
    plt.tight_layout()

    # Save SVG
    plt.savefig(output_svg, format="svg")
    print(f"Saved SVG → {output_svg}")

    plt.show()


if __name__ == "__main__":
    main()

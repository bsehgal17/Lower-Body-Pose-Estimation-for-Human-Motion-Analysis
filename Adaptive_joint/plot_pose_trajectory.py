import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- USER SETTINGS ---
json_files = [
    "pose_file1.json",
    "pose_file2.json",
    "pose_file3.json"
]
joint_index = 11  # the joint number you want to extract
# ---------------------

def load_pose(json_path):
    with open(json_path, "r") as f:
        return json.load(f)

def extract_joint_trajectory(pose_data, joint_index):
    trajectory = []

    # detect frames
    if "frames" in pose_data:
        frames = pose_data["frames"]
    elif "keypoints" in pose_data:
        frames = pose_data["keypoints"]
    else:
        return np.empty((0, 2))

    for frame in frames:
        kps = frame["keypoints"] if isinstance(frame, dict) else frame

        # unwrap [[...]] → [...]
        if isinstance(kps, list) and len(kps) == 1 and isinstance(kps[0], list):
            kps = kps[0]

        if len(kps) <= joint_index:
            continue

        kp = kps[joint_index]
        if not isinstance(kp, (list, tuple)) or len(kp) < 2:
            continue

        x, y = float(kp[0]), float(kp[1])
        if not (np.isnan(x) or np.isnan(y)):
            trajectory.append([x, y])

    return np.asarray(trajectory)

def main():
    plt.figure(figsize=(6, 6))

    for json_file in json_files:
        pose = load_pose(json_file)
        traj = extract_joint_trajectory(pose, joint_index)

        if traj.shape[0] == 0:
            print(f"⚠ No data in {json_file}")
            continue

        plt.plot(
            traj[:, 0],
            traj[:, 1],
            label=Path(json_file).stem,
            linewidth=2,
        )

    plt.gca().invert_yaxis()  # image coordinate system
    plt.axis("equal")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"Joint {joint_index} Trajectory")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

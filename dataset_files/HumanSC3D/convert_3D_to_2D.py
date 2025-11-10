import os
import json
import numpy as np
import cv2


# ---------- Utility functions ----------


def read_video(vid_path):
    frames = []
    cap = cv2.VideoCapture(vid_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.array(frames)


def read_cam_params(cam_path):
    with open(cam_path) as f:
        cam_params = json.load(f)
    for key1 in cam_params:
        for key2 in cam_params[key1]:
            cam_params[key1][key2] = np.array(cam_params[key1][key2])
    return cam_params


def read_data(
    data_root, dataset_name, train_folder, subset, subj_name, action_name, camera_name
):
    try:
        # Paths
        base_path = f"{data_root}/{dataset_name}/{train_folder}/{subset}/{subj_name}"
        vid_path = f"{base_path}/videos/{camera_name}/{action_name}.mp4"
        cam_path = f"{base_path}/camera_parameters/{camera_name}/{action_name}.json"
        j3d_path = f"{base_path}/joints3d_25/{action_name}.json"

        if (
            not os.path.exists(vid_path)
            or not os.path.exists(j3d_path)
            or not os.path.exists(cam_path)
        ):
            return None

        frames = read_video(vid_path)
        cam_params = read_cam_params(cam_path)
        with open(j3d_path) as f:
            j3ds = np.array(json.load(f)["joints3d_25"])

        seq_len = min(len(frames), j3ds.shape[0])
        return frames[:seq_len], j3ds[:seq_len], cam_params
    except Exception as e:
        print(f"⚠️ Skipping {subj_name} {action_name} {camera_name} due to error: {e}")
        return None


def project_3d_to_2d(points3d, intrinsics, intrinsics_type):
    if intrinsics_type == "w_distortion":
        p = intrinsics["p"][:, [1, 0]]
        x = points3d[:, :2] / points3d[:, 2:3]
        r2 = np.sum(x**2, axis=1)
        radial = 1 + np.transpose(
            np.matmul(intrinsics["k"], np.array([r2, r2**2, r2**3]))
        )
        tan = np.matmul(x, np.transpose(p))
        xx = x * (tan + radial) + r2[:, np.newaxis] * p
        proj = intrinsics["f"] * xx + intrinsics["c"]
    else:
        xx = points3d[:, :2] / points3d[:, 2:3]
        proj = intrinsics["f"] * xx + intrinsics["c"]
    return proj


def draw_pose(frame, points_2d):
    limbs = [
        [10, 9],
        [9, 8],
        [8, 11],
        [8, 14],
        [11, 12],
        [14, 15],
        [12, 13],
        [15, 16],
        [8, 7],
        [7, 0],
        [0, 1],
        [0, 4],
        [1, 2],
        [4, 5],
        [2, 3],
        [5, 6],
        [13, 21],
        [13, 22],
        [16, 23],
        [16, 24],
        [3, 17],
        [3, 18],
        [6, 19],
        [6, 20],
    ]
    frame_out = frame.copy()
    for limb in limbs:
        if limb[0] < len(points_2d) and limb[1] < len(points_2d):
            pt1 = tuple(points_2d[limb[0]].astype(int))
            pt2 = tuple(points_2d[limb[1]].astype(int))
            cv2.line(frame_out, pt1, pt2, (0, 255, 0), 2)
    for x, y in points_2d:
        cv2.circle(frame_out, (int(x), int(y)), 3, (0, 0, 255), -1)
    return frame_out


# ---------- Main pipeline ----------

if __name__ == "__main__":
    dataset_name = "humansc3d"
    data_root = r"C:/Users/BhavyaSehgal/Downloads"
    train_folder = "humansc3d_train"
    subset = "train"

    subjects_path = os.path.join(data_root, dataset_name, train_folder, subset)
    subjects = [s for s in os.listdir(subjects_path) if s.startswith("s")]

    for subj_name in subjects:
        print(f"\n📂 Processing subject: {subj_name}")
        subj_path = os.path.join(subjects_path, subj_name)
        videos_path = os.path.join(subj_path, "videos")
        if not os.path.exists(videos_path):
            continue

        cameras = os.listdir(videos_path)

        # create per-subject output folders
        output_root = os.path.join(subj_path, "processed_outputs")
        videos_out_dir = os.path.join(output_root, "videos")
        points_out_dir = os.path.join(output_root, "2d_points")
        os.makedirs(videos_out_dir, exist_ok=True)
        os.makedirs(points_out_dir, exist_ok=True)

        for camera_name in cameras:
            cam_dir = os.path.join(videos_path, camera_name)
            actions = [
                a.replace(".mp4", "") for a in os.listdir(cam_dir) if a.endswith(".mp4")
            ]

            for action_name in actions:
                print(f"→ {action_name} | {camera_name}")
                data = read_data(
                    data_root,
                    dataset_name,
                    train_folder,
                    subset,
                    subj_name,
                    action_name,
                    camera_name,
                )
                if data is None:
                    continue

                frames, j3ds, cam_params = data

                # Output paths
                video_out = os.path.join(
                    videos_out_dir, f"{action_name}_{camera_name}.mp4"
                )
                json_out = os.path.join(
                    points_out_dir, f"{action_name}_{camera_name}_2d.json"
                )

                height, width = frames[0].shape[:2]
                writer = cv2.VideoWriter(
                    video_out, cv2.VideoWriter_fourcc(*"mp4v"), 30, (width, height)
                )
                all_2d_points = []

                for i in range(len(frames)):
                    j3d = j3ds[i]
                    j3d_in_camera = np.matmul(
                        np.array(j3d) - cam_params["extrinsics"]["T"],
                        cam_params["extrinsics"]["R"].T,
                    )
                    j2d = project_3d_to_2d(
                        j3d_in_camera,
                        cam_params["intrinsics_w_distortion"],
                        "w_distortion",
                    )
                    all_2d_points.append(j2d.tolist())

                    overlay = draw_pose(cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR), j2d)
                    writer.write(overlay)
                writer.release()

                # Save JSON
                with open(json_out, "w") as f:
                    json.dump({"2d_keypoints": all_2d_points}, f, indent=2)

                print(f"Saved video: {video_out}")
                print(f"Saved points: {json_out}")

    print("\nAll subjects processed successfully!")

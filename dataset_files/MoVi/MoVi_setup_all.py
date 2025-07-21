import os
import cv2
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt


def load_camera_matrix_npz(npz_path):
    data = np.load(npz_path)
    K = data["IntrinsicMatrix"].T.astype(np.float32)
    dist = np.zeros((5, 1), dtype=np.float32)
    if "RadialDistortion" in data:
        rad = data["RadialDistortion"]
        dist[: len(rad), 0] = rad
    if "TangentialDistortion" in data:
        tan = data["TangentialDistortion"]
        dist[2 : 2 + len(tan), 0] = tan
    return K, dist


def load_extrinsics_npz(npz_path):
    data = np.load(npz_path)
    R = data["rotationMatrix"].astype(np.float32)
    T = data["translationVector"].reshape(3, 1).astype(np.float32)
    return R, T


def extract_amass_joints(amass_path, motion="walking"):
    mat = sio.loadmat(amass_path, struct_as_record=False, squeeze_me=True)
    subject_key = [k for k in mat.keys() if k.startswith("Subject")][0]
    subject = mat[subject_key]

    motions_list = []
    valid_motions = []
    for m in subject.move:
        try:
            if hasattr(m, "description") and isinstance(m.description, str):
                motions_list.append(m.description)
                valid_motions.append(m)
        except Exception:
            continue

    if motion not in motions_list:
        raise ValueError(
            f"Motion '{motion}' not found in {amass_path}. Available motions: {motions_list}"
        )

    motion_idx = motions_list.index(motion)
    selected_motion = valid_motions[motion_idx]

    return selected_motion.jointsLocation_amass, selected_motion


def read_entire_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return np.array(frames)


def extract_video_segment(video_path, output_path, start_sec, duration_sec):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_frame = int(start_sec * fps)
    end_frame = min(int((start_sec + duration_sec) * fps), total_frames)

    out = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"XVID"), fps, (width, height)
    )
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for _ in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    cap.release()
    out.release()


def convert_world_points_to_image_points(K, R, T, world_points):
    translation_vector_expand = T.T
    rot_tran_matrix = np.concatenate((R, translation_vector_expand), axis=0)
    camera_matrix = np.dot(K, rot_tran_matrix.T).T

    image_points = np.zeros((world_points.shape[0], 2), dtype=np.float32)

    for idx, val in enumerate(world_points):
        temp_matrix = np.append(val, 1)
        result = np.dot(temp_matrix, camera_matrix)
        u = result[0] / result[2]
        v = result[1] / result[2]
        image_points[idx, :] = [u, v]

    return image_points


def project_3d_to_2d(joints3d, K, R, T, fps=30, mocap_fps=120):
    downsample = int(mocap_fps / fps)
    joints_ds = joints3d[::downsample]
    projected = np.zeros(
        (joints_ds.shape[0], joints_ds.shape[1], 2), dtype=np.float32
    )  # <== MODIFIED
    for i, frame in enumerate(joints_ds):
        projected[i] = convert_world_points_to_image_points(K, R, T, frame)
    return projected


def parse_v3d_segments(v3d_path):
    mat = sio.loadmat(v3d_path, struct_as_record=False, squeeze_me=True)
    subject_key = [k for k in mat if k.startswith("Subject")][0]
    move = mat[subject_key].move
    flags30 = move.flags30
    motions = move.motions_list
    return {
        m.lower(): (int(start), int(end)) for m, (start, end) in zip(motions, flags30)
    }


def overlay_video_with_joints(frames, joints_2d, save_path):
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    height, width = frames[0].shape[:2]
    out = cv2.VideoWriter(save_path, fourcc, 30, (width, height))

    for frame, joints in zip(frames, joints_2d):
        for j, (x, y) in enumerate(joints):
            if 0 <= x < width and 0 <= y < height:
                cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)
        out.write(frame)
    out.release()


def process_all_subjects(
    intr_dir, extr_dir, amass_dir, video_dir, v3d_dir, output_root, motion="walking"
):
    os.makedirs(output_root, exist_ok=True)
    subject_ids = [
        f.split("_")[-1].split(".")[0]
        for f in os.listdir(amass_dir)
        if f.endswith(".mat")
    ]

    for sid in subject_ids:
        try:
            video_path = os.path.join(video_dir, f"F_PG1_Subject_{sid}_L.avi")
            if not os.path.exists(video_path):
                print(f"❌ Skipping Subject {sid}: video not found.")
                continue

            print(f"\n=== Processing Subject {sid} ===")

            intr_path = os.path.join(intr_dir, "cameraParams_PG1.npz")
            extr_path = os.path.join(extr_dir, "Extrinsics_PG1.npz")
            amass_path = os.path.join(amass_dir, f"F_amass_Subject_{sid}.mat")
            v3d_path = os.path.join(v3d_dir, f"F_v3d_Subject_{sid}.mat")

            K, _ = load_camera_matrix_npz(intr_path)
            R, T = load_extrinsics_npz(extr_path)
            joints3d, _ = extract_amass_joints(amass_path, motion)
            fps = cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FPS)
            segments = parse_v3d_segments(v3d_path)

            if motion.lower() not in segments:
                print(f"⚠ Skipping: '{motion}' not in V3D segments for Subject {sid}")
                continue

            subj_out = os.path.join(output_root, f"Subject_{sid}")
            os.makedirs(subj_out, exist_ok=True)

            start_frame, end_frame = segments[motion.lower()]
            start_sec = start_frame / fps
            duration_sec = (end_frame - start_frame) / fps

            cropped_video_path = os.path.join(subj_out, "walking_cropped.avi")
            extract_video_segment(
                video_path, cropped_video_path, start_sec, duration_sec
            )

            video_frames = read_entire_video(cropped_video_path)
            joints2d = project_3d_to_2d(joints3d, K, R, T, fps=int(fps))
            joints2d_trimmed = joints2d[: len(video_frames)]
            np.save(os.path.join(subj_out, "joints2d_projected.npy"), joints2d_trimmed)

            overlay_path = os.path.join(subj_out, "walking_overlay.avi")
            overlay_video_with_joints(video_frames, joints2d_trimmed, overlay_path)

            print(f"✔ Subject {sid} processed: cropped, overlay, and joints saved.\n")

        except Exception as e:
            print(f"⚠ Error processing Subject {sid}: {e}")


# === Entry Point ===
if __name__ == "__main__":
    base = r"C:\Users\BhavyaSehgal\Downloads\MoVi dataset"
    process_all_subjects(
        intr_dir=os.path.join(base, "Camera Parameters", "Calib"),
        extr_dir=os.path.join(base, "Camera Parameters", "Calib"),
        amass_dir=os.path.join(base, "AMASS"),
        video_dir=os.path.join(base, "dataverse_files"),
        v3d_dir=os.path.join(base, "MoVi_mocap"),
        output_root=os.path.join(base, "results_all_subjects"),
        motion="walking",
    )

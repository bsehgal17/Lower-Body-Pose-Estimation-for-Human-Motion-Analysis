import numpy as np
import pandas as pd
import os

npz_file = r"C:\Users\BhavyaSehgal\Downloads\data_2d_humaneva20_gt.npz"
output_folder = r"C:\Users\BhavyaSehgal\Downloads\output_csvs"
os.makedirs(output_folder, exist_ok=True)

data = np.load(npz_file, allow_pickle=True)
data_dict = data["positions_2d"].item()  # unwrap dictionary

for subject, chunks in data_dict.items():
    subject_folder = os.path.join(output_folder, subject.replace("/", "_"))
    os.makedirs(subject_folder, exist_ok=True)

    for chunk_name, array_list in chunks.items():
        array_list = np.array(array_list)  # convert list to np.array if needed
        combined_rows = []

        # Determine number of cameras
        if array_list.ndim == 4 or isinstance(array_list, list):
            num_cameras = (
                len(array_list) if isinstance(array_list, list) else array_list.shape[0]
            )

            for cam_idx in range(num_cameras):
                cam_array = (
                    array_list[cam_idx]
                    if isinstance(array_list, list)
                    else array_list[cam_idx]
                )

                if cam_array.ndim != 3:
                    print(
                        f"⚠️ Skipping {chunk_name} in {subject}, unexpected shape: {cam_array.shape}"
                    )
                    continue

                frames, joints, coords = cam_array.shape
                flattened = cam_array.reshape(frames, joints * coords)

                # Add Camera column
                camera_col = np.full((frames, 1), cam_idx + 1)
                combined_rows.append(
                    np.hstack([camera_col, flattened[:, : joints * 2]])
                )

        elif array_list.ndim == 3:  # single camera
            frames, joints, coords = array_list.shape
            flattened = array_list.reshape(frames, joints * coords)
            camera_col = np.ones((frames, 1))
            combined_rows.append(np.hstack([camera_col, flattened[:, : joints * 2]]))
        else:
            print(
                f"⚠️ Skipping {chunk_name} in {subject}, unexpected shape: {array_list.shape}"
            )
            continue

        # Combine all camera rows
        all_data = np.vstack(combined_rows)
        # Interleave x and y columns for each joint from [x1, y1, x2, y2, ...]
        interleaved = []
        for row in all_data:
            camera = row[0]
            coords = row[1:]
            # Each joint has 2 values: x, y
            xy_pairs = [coords[2 * j : 2 * j + 2] for j in range(joints)]
            flat_xy = [val for pair in xy_pairs for val in pair]
            interleaved.append([camera] + flat_xy)
        # New column order: Camera, x1, y1, x2, y2, ...
        new_columns = ["Camera"] + [
            f"{axis}{j + 1}" for j in range(joints) for axis in ("x", "y")
        ]
        df_xy = pd.DataFrame(interleaved, columns=new_columns)

        # Add Subject and Action columns
        df_xy.insert(0, "Action", chunk_name)
        df_xy.insert(0, "Subject", subject)

        # Save CSV
        chunk_name_clean = chunk_name.replace(" ", "_")
        csv_path = os.path.join(subject_folder, f"{chunk_name_clean}.csv")
        df_xy.to_csv(csv_path, index=False)
        print(f"Saved {csv_path}")

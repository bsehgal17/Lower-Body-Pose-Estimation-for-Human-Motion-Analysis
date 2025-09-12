import numpy as np
import pandas as pd
import os

# Path to your .npy file
npy_file = r"C:\Users\BhavyaSehgal\Downloads\positions_3d.npy"
output_folder = r"C:\Users\BhavyaSehgal\Downloads\output_csvs"
os.makedirs(output_folder, exist_ok=True)

# Load .npy file
data = np.load(npy_file, allow_pickle=True)
data_dict = data.item()  # extract dictionary

for subject, chunks in data_dict.items():
    subject_folder = os.path.join(output_folder, subject.replace("/", "_"))
    os.makedirs(subject_folder, exist_ok=True)

    for chunk_name, array_3d in chunks.items():
        frames, joints, coords = array_3d.shape

        # Flatten joints to x1,y1,x2,y2,... format
        flattened = array_3d.reshape(frames, joints * coords)
        # Create columns: x1,y1,x2,y2,...
        columns = []
        for j in range(joints):
            columns += [f"x{j + 1}", f"y{j + 1}"]  # only x and y
        df_xy = pd.DataFrame(flattened[:, : joints * 2], columns=columns)

        # Add Subject, Action, Camera columns
        df_xy.insert(0, "Camera", 0)
        df_xy.insert(0, "Action", chunk_name)
        df_xy.insert(0, "Subject", subject)

        # Save CSV
        chunk_name_clean = chunk_name.replace(" ", "_")
        csv_path = os.path.join(subject_folder, f"{chunk_name_clean}.csv")
        df_xy.to_csv(csv_path, index=False)
        print(f"Saved {csv_path}")

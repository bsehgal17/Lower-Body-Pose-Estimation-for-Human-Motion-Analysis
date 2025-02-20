import numpy as np
import scipy.io


def extract_ground_truth(mat_file):
    print("Loading MAT file (Ground Truth)...")
    mat_data = scipy.io.loadmat(mat_file)
    markers = mat_data["Markers"]  # Shape: (frames, joints, 3)
    print(f"Loaded MAT file with shape: {markers.shape}")

    # Extract labels from ParameterGroup(1,8)
    event_data = mat_data["ParameterGroup"][0][2]  # Access ParameterGroup(1,8)
    labels = []

    for entry in event_data[2][0]:  # Iterate over all parameter entries
        name = entry[0][0]  # Extract name
        if name == "LABELS":
            labels_array = entry[2]  # Extract actual label names
            # Convert to list of strings
            labels = [str(label[0]) for label in labels_array[0]]
            break

    # Map MAT joint names to keypoint labels
    mat_to_halpe_mapping = {
        "RL:LASI": "left_hip", "RL:RASI": "right_hip",
        "RL:LKNE": "left_knee", "RL:RKNE": "right_knee",
        "RL:LANK": "left_ankle", "RL:RANK": "right_ankle",
        "RL:LTOE": "left_big_toe", "RL:RTOE": "right_big_toe",
        "RL:LHEE": "left_heel", "RL:RHEE": "right_heel"
    }

    frames = markers.shape[0]
    gt_keypoints = []

    for i in range(frames):
        gt_frame = []
        for mat_joint, halpe_joint in mat_to_halpe_mapping.items():
            if mat_joint in labels:
                idx = labels.index(mat_joint)
                gt_point = markers[i, idx, :2]  # Use only X, Y
                gt_frame.append(gt_point)

        gt_keypoints.append(np.array(gt_frame))

    return np.array(gt_keypoints)

import numpy as np
import json


def extract_predictions(json_file):
    print("Loading JSON file (Predictions)...")
    with open(json_file, "r") as f:
        json_data = json.load(f)
    print(f"Loaded JSON file with {len(json_data)} frames")

    halpe_joint_indices = {11: "left_hip", 12: "right_hip", 13: "left_knee", 14: "right_knee",
                           15: "left_ankle", 16: "right_ankle", 20: "left_big_toe", 21: "right_big_toe",
                           24: "left_heel", 25: "right_heel"}
    halpe_joint_name_to_index = {
        name: index for index, name in halpe_joint_indices.items()}

    frames = len(json_data)
    pred_keypoints = []

    for i in range(frames):
        pred_frame = []
        json_kpts = np.array(
            json_data[i]["keypoints"]).reshape(-1, 3)[:, :2]  # Only X, Y

        for halpe_joint, halpe_index in halpe_joint_name_to_index.items():
            pred_point = json_kpts[halpe_index]
            pred_frame.append(pred_point)

        pred_keypoints.append(np.array(pred_frame))

    return np.array(pred_keypoints)

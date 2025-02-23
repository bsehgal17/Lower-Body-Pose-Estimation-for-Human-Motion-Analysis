import numpy as np
import json


def extract_predictions(json_file, frame_range):
    print("Loading JSON file (Predictions)...")
    with open(json_file, "r") as f:
        json_data = json.load(f)
    print(f"Loaded JSON file with {len(json_data)} frames")

    coco_joints = {'left_hip': 12, 'right_hip': 13, 'left_knee': 13, 'right_knee': 15, 'left_ankle': 16, 'right_ankle': 17}

    
    pred_keypoints = []

    frame_end = frame_range[1] if len(json_data)>frame_range[1] else len(json_data)
    for i in range(frame_range[0],frame_end):
        pred_frame = []
        json_kpts = np.array(
            json_data[i]["keypoints"][0]["keypoints"][0]) 

        for joint,index in coco_joints.items():
            pred_point = json_kpts[index]
            pred_frame.append(pred_point)

        pred_keypoints.append(np.array(pred_frame))

    return np.array(pred_keypoints)

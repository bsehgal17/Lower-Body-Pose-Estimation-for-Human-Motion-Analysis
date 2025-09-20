import os
import json
import pickle
import numpy as np


def convert_alphapose_to_pkl(json_path, output_pkl_path):
    """Convert a single AlphaPose JSON file to HumanEva-compatible .pkl format."""
    with open(json_path, "r") as f:
        data = json.load(f)

    persons = {}
    for det in data:
        # Extract frame index from image_id (last part before .jpg)
        frame_str = det["image_id"].split("_")[-1].replace(".jpg", "")
        frame_idx = int(frame_str) - 1  # zero-based index

        keypoints = np.array(det["keypoints"]).reshape(-1, 3).tolist()
        bbox = det.get("bbox", [0, 0, 0, 0])
        score = det.get("score", 0.0)
        visibility = [kp[2] for kp in keypoints]

        person_id = 0  # assuming one main person, else assign different IDs if needed

        if person_id not in persons:
            persons[person_id] = {"person_id": person_id, "poses": [], "detections": []}

        persons[person_id]["poses"].append(
            {
                "frame_idx": frame_idx,
                "keypoints": keypoints,
                "bbox": bbox,
                "keypoints_visible": visibility,
            }
        )

        persons[person_id]["detections"].append(
            {"frame_idx": frame_idx, "score": score}
        )

    pred_data = {"persons": list(persons.values())}

    os.makedirs(os.path.dirname(output_pkl_path), exist_ok=True)
    with open(output_pkl_path, "wb") as f:
        pickle.dump(pred_data, f)

    print(f"✅ Converted {json_path} → {output_pkl_path}")


def batch_convert_jsons(input_folder, output_folder):
    """Convert all AlphaPose JSON files in a folder to .pkl format."""
    os.makedirs(output_folder, exist_ok=True)

    for file_name in os.listdir(input_folder):
        if file_name.endswith(".json"):
            json_path = os.path.join(input_folder, file_name)
            base_name = os.path.splitext(file_name)[0]
            output_pkl_path = os.path.join(output_folder, base_name + ".pkl")

            convert_alphapose_to_pkl(json_path, output_pkl_path)


if __name__ == "__main__":
    input_folder = "/path/to/alphapose/jsons"  # 🔹 folder with your AlphaPose JSONs
    output_folder = "/path/to/converted/pkls"  # 🔹 folder where PKLs should be saved
    batch_convert_jsons(input_folder, output_folder)

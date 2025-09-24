import json
import pickle
import os
import numpy as np


def convert_mediapipe_json_to_standard(input_json_path, output_dir=None):
    with open(input_json_path, "r") as f:
        data = json.load(f)

    frames = data["frames"]
    persons = []
    person_id = 0  # assume single person

    poses = []
    detections = []

    for frame in frames:
        frame_idx = frame["frame_number"]
        landmarks = frame["landmarks"]

        # Extract keypoints (x,y) and confidence (visibility)
        keypoints = []
        visibility = []
        xs, ys = [], []

        for lm in landmarks:
            x, y, vis = lm["x"], lm["y"], lm["visibility"]
            keypoints.append([x, y, vis])
            visibility.append(vis)
            xs.append(x)
            ys.append(y)

        keypoints = np.array(keypoints)

        # Simple bbox around landmarks
        if len(xs) > 0:
            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)
            bbox = [xmin, ymin, xmax, ymax]
        else:
            bbox = [0, 0, 0, 0]

        score = float(np.mean(visibility)) if len(visibility) > 0 else 0.0

        poses.append(
            {
                "frame_idx": frame_idx,
                "keypoints": keypoints.tolist(),
                "keypoints_visible": visibility,
                "bbox": bbox,
                "bbox_scores": [score],
            }
        )

        detections.append({"frame_idx": frame_idx, "bbox": bbox, "score": score})

    persons.append({"person_id": person_id, "poses": poses, "detections": detections})

    standard_dict = {"persons": persons}

    # Output paths
    base_name = os.path.splitext(os.path.basename(input_json_path))[0]
    if output_dir is None:
        output_dir = os.path.dirname(input_json_path)

    out_json_path = os.path.join(output_dir, f"{base_name}_standard.json")
    out_pkl_path = os.path.join(output_dir, f"{base_name}_standard.pkl")

    # Save JSON
    with open(out_json_path, "w") as f:
        json.dump(standard_dict, f, indent=2)

    # Save PKL
    with open(out_pkl_path, "wb") as f:
        pickle.dump(standard_dict, f)

    print(f"Saved standardized JSON: {out_json_path}")
    print(f"Saved standardized PKL:  {out_pkl_path}")


def process_folder(root_folder):
    """Recursively process all JSON files in a folder and subfolders."""
    for dirpath, _, filenames in os.walk(root_folder):
        for fname in filenames:
            if fname.endswith(".json") and not fname.endswith("_standard.json"):
                input_path = os.path.join(dirpath, fname)
                try:
                    convert_mediapipe_json_to_standard(input_path, dirpath)
                except Exception as e:
                    print(f"Failed to process {input_path}: {e}")


# ---------------------------
# Run conversion when script is executed
# ---------------------------
if __name__ == "__main__":
    folder_path = r"C:\Users\BhavyaSehgal\Downloads\Mediapipe\pose_data"  # Change this to your folder path
    process_folder(folder_path)

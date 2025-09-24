import json
import pickle
import os
import numpy as np
import math


def convert_mediapipe_json_to_standard(input_json_path, output_dir):
    with open(input_json_path, "r") as f:
        data = json.load(f)

    frames = data.get("frames", [])
    persons = []
    person_id = 0  # assume single person

    poses = []
    detections = []

    for frame in frames:
        frame_idx = frame.get("frame_number", -1)
        landmarks = frame.get("landmarks", [])

        # Get frame size (replace with actual values if not present in JSON)
        frame_width = data.get("resolution")[0]   # default width
        frame_height = data.get("resolution")[1]  # default height

        if landmarks and len(landmarks) > 0:
            keypoints = []
            visibility = []
            xs, ys = [], []

            for lm in landmarks:
                # Convert normalized coordinates to pixel coordinates
                x = min(math.floor(lm.get("x", 0.0) *
                        frame_width), frame_width - 1)
                y = min(math.floor(lm.get("y", 0.0) *
                        frame_height), frame_height - 1)
                vis = lm.get("visibility", 0.0)

                keypoints.append([x, y])
                visibility.append(vis)
                xs.append(x)
                ys.append(y)

            keypoints = np.array(keypoints)

            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)
            bbox = [xmin, ymin, xmax, ymax]
            score = float(np.mean(visibility)) if len(visibility) > 0 else 0.0
        else:
            # No detection in this frame
            keypoints = None
            visibility = []
            bbox = [0, 0, 0, 0]
            score = 0.0

        poses.append(
            {
                "frame_idx": frame_idx,
                "keypoints": keypoints.tolist() if keypoints is not None else None,
                "keypoints_visible": visibility,
                "bbox": bbox,
                "bbox_scores": [score],
            }
        )

        detections.append(
            {"frame_idx": frame_idx, "bbox": bbox, "score": score}
        )

    persons.append({"person_id": person_id, "poses": poses,
                   "detections": detections})
    standard_dict = {"persons": persons}

    # Output paths
    base_name = os.path.splitext(os.path.basename(input_json_path))[0]
    os.makedirs(output_dir, exist_ok=True)

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

    # Create output root folder parallel to the input root folder
    parent_dir = os.path.dirname(root_folder)
    output_root = os.path.join(parent_dir, "detect/standardized_results")
    os.makedirs(output_root, exist_ok=True)

    for dirpath, _, filenames in os.walk(root_folder):
        # Keep subfolder structure inside output folder
        rel_path = os.path.relpath(dirpath, root_folder)
        output_dir = os.path.join(output_root, rel_path)
        os.makedirs(output_dir, exist_ok=True)

        for fname in filenames:
            if fname.endswith(".json") and not fname.endswith("_standard.json"):
                input_path = os.path.join(dirpath, fname)
                try:
                    convert_mediapipe_json_to_standard(input_path, output_dir)
                except Exception as e:
                    print(f"Failed to process {input_path}: {e}")


# ---------------------------
# Run conversion when script is executed
# ---------------------------
if __name__ == "__main__":
    folder_path = "/storage/Projects/Gaitly/bsehgal/lower_body_pose_est/pipeline_results/HumanEva/Mediapipe/pose_data"
    process_folder(folder_path)

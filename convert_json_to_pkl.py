import json
import pickle
from pathlib import Path


def json_to_pkl_recursive(
    root_dir: str,
    overwrite: bool = False,
):
    """
    Recursively find all .json files and save corresponding .pkl files
    in the same directory.

    Args:
        root_dir (str): Root directory to search.
        overwrite (bool): Overwrite existing .pkl files if True.
    """

    root = Path(root_dir)

    for json_path in root.rglob("*.json"):
        pkl_path = json_path.with_suffix(".pkl")

        if pkl_path.exists() and not overwrite:
            print(f"Skipping (exists): {pkl_path}")
            continue

        try:
            with open(json_path, "r") as f:
                data = json.load(f)

            with open(pkl_path, "wb") as f:
                pickle.dump(data, f)

            print(f"Saved: {pkl_path}")

        except Exception as e:
            print(f"Failed to convert {json_path}: {e}")


json_to_pkl_recursive(
    root_dir="/storage/Projects/Gaitly/bsehgal/lower_body_pose_est/pipeline_results/Adaptive_filt/adaptive_filtering/",
    overwrite=False,  # set True if you want to replace existing .pkl files
)

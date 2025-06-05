# processing/filters_runner.py
import json
import os
import logging
from typing import Dict, Any, List
import numpy as np

from config.base import GlobalConfig
from post_processing.clean_data import interpolate_series
from post_processing.filters import (
    butterworth_filter,
    gaussian_filter,
    median_filter_1d,
    kalman_filter,
    kalman_rts_filter,
    moving_average_filter,
    savitzky_golay_filter,
    wiener_filter_1d,
)
from utils.video_info import extract_video_info
from utils.joint_enum import PredJoints  # Ensure this enum is correctly defined
from utils.utils import plot_filtering_effect  # Make sure this utility exists

logger = logging.getLogger(__name__)

# Dispatch map for filter functions
FILTER_FN_MAP = {
    "gaussian": gaussian_filter,
    "butterworth": butterworth_filter,
    "median": median_filter_1d,
    "kalman": kalman_filter,
    "kalman_rts": kalman_rts_filter,
    "moving_average": moving_average_filter,
    "savitzky": savitzky_golay_filter,
    "wiener": wiener_filter_1d,
}


def save_filtered_keypoints(
    output_folder: str,
    original_json_path: str,
    filtered_keypoints: List[Dict],
    filter_name: str,
):
    """Saves the filtered keypoints to a new JSON file."""
    os.makedirs(output_folder, exist_ok=True)
    filtered_json_path = os.path.join(
        output_folder,
        os.path.basename(original_json_path).replace(
            ".json", f"_{filter_name}_filtered.json"
        ),
    )
    with open(filtered_json_path, "w") as f:
        json.dump(filtered_keypoints, f, indent=4)
    logger.info(f"Filtered keypoints saved to: {filtered_json_path}")


def run_keypoint_filtering(
    config: GlobalConfig, filter_name: str, filter_kwargs: Dict[str, Any]
):
    """
    Applies a specified filter to keypoint data from detected poses.

    Args:
        config (GlobalConfig): The global configuration object.
        filter_name (str): The name of the filter to apply.
        filter_kwargs (Dict[str, Any]): Keyword arguments for the filter function.
    """
    logger.info(
        f"Starting keypoint filtering with filter: '{filter_name}' and kwargs: {filter_kwargs}"
    )

    # Use config for base paths and parameters
    input_json_base_dir = (
        config.paths.output_dir
    )  # Assuming detected JSONs are in output_dir
    filtered_output_base_dir = os.path.join(
        config.paths.output_dir, "filtered_poses"
    )  # New dir for filtered data

    iqr_multiplier = config.processing.get(
        "iqr_multiplier", 1.5
    )  # Add to config if not there
    interpolation_kind = config.processing.get(
        "interpolation_kind", "linear"
    )  # Add to config

    # Define joints to filter (these could also be configured)
    joints_to_filter = [
        PredJoints.LEFT_ANKLE.value,
        PredJoints.RIGHT_ANKLE.value,
        PredJoints.LEFT_HIP.value,
        PredJoints.RIGHT_HIP.value,
        PredJoints.LEFT_KNEE.value,
        PredJoints.RIGHT_KNEE.value,
    ]

    if filter_name not in FILTER_FN_MAP:
        logger.error(
            f"Unknown filter: {filter_name}. Available filters: {list(FILTER_FN_MAP.keys())}"
        )
        return

    filter_fn = FILTER_FN_MAP[filter_name]

    # Iterate through the pose estimation results (JSON files)
    # This logic assumes the JSON files are structured in a predictable way
    # within config.paths.output_dir

    # You'll need to adapt this `os.walk` part based on how your `detect_and_visualize_pose`
    # saves its JSONs. If it saves them as `output_dir/subject/action/camera/video_name.json`
    # then you need to walk that structure.

    for root, _, files in os.walk(input_json_base_dir):
        for file in files:
            if file.endswith(".json") and not file.endswith(
                "_filtered.json"
            ):  # Avoid reprocessing filtered files
                json_path = os.path.join(root, file)

                # Derive video info from the JSON path if possible, or assume a flat structure
                # This part is highly dependent on your saving structure.
                # Example: if your JSONs are directly in `output_dir/video_name.json`
                # Or if your JSONs are in `output_dir/subject/action/camera/video_name.json`

                # For this example, let's assume `extract_video_info` can work from JSON path
                # or you have a helper that maps JSON path back to video info.
                video_info = extract_video_info(
                    file, root
                )  # Needs adaptation if it only takes video name
                if not video_info:
                    logger.warning(
                        f"Could not extract video info from {json_path}. Skipping."
                    )
                    continue
                subject, action, camera_idx = (
                    video_info  # Adjust according to extract_video_info output
                )
                action_group = action.replace(" ", "_")

                logger.info(f"Applying filter to {json_path}")

                try:
                    with open(json_path, "r") as f:
                        pred_keypoints = json.load(f)

                    # Create a deep copy to modify and save as filtered
                    filtered_pred_keypoints = json.loads(json.dumps(pred_keypoints))

                    # Iterate over each person's keypoints (if multiple detected)
                    # And then over the specific joints you want to filter
                    for frame_idx, frame_data in enumerate(filtered_pred_keypoints):
                        for person_idx, person_data in enumerate(
                            frame_data.get("keypoints", [])
                        ):
                            kp_series = {}
                            for joint_id in joints_to_filter:
                                kp = person_data.get("keypoints", {}).get(
                                    joint_id
                                )  # Assuming kp format {id: [x,y,c]}
                                if kp:
                                    if joint_id not in kp_series:
                                        kp_series[joint_id] = {"x": [], "y": []}
                                    kp_series[joint_id]["x"].append(kp[0])
                                    kp_series[joint_id]["y"].append(kp[1])
                                else:
                                    kp_series[joint_id]["x"].append(
                                        float("nan")
                                    )  # Append NaN for missing data
                                    kp_series[joint_id]["y"].append(float("nan"))

                            for joint_id in joints_to_filter:
                                x_series = kp_series[joint_id]["x"]
                                y_series = kp_series[joint_id]["y"]

                                if all(np.isnan(x_series)) or all(np.isnan(y_series)):
                                    logger.warning(
                                        f"Joint {joint_id} has all NaN values for person {person_idx} in {file}. Skipping filter."
                                    )
                                    continue

                                try:
                                    x_interp = interpolate_series(
                                        x_series, iqr_multiplier, interpolation_kind
                                    )
                                    y_interp = interpolate_series(
                                        y_series, iqr_multiplier, interpolation_kind
                                    )

                                    x_filtered = filter_fn(x_interp, **filter_kwargs)
                                    y_filtered = filter_fn(y_interp, **filter_kwargs)

                                    # Update the filtered_pred_keypoints
                                    for i, frame in enumerate(filtered_pred_keypoints):
                                        if frame_data.get("keypoints"):
                                            # This logic needs to correctly map filtered values back to the structure
                                            # This assumes a structure where `keypoints` is a list of person dicts,
                                            # and each person dict has a `keypoints` dict mapping ID to [x,y,c]
                                            # Adapt this part based on `combine_keypoints` output
                                            if (
                                                person_data.get("keypoints")
                                                and joint_id
                                                in frame["keypoints"][person_idx][
                                                    "keypoints"
                                                ]
                                            ):
                                                frame["keypoints"][person_idx][
                                                    "keypoints"
                                                ][joint_id][0] = float(x_filtered[i])
                                                frame["keypoints"][person_idx][
                                                    "keypoints"
                                                ][joint_id][1] = float(y_filtered[i])

                                except Exception as e:
                                    logger.warning(
                                        f"Error filtering joint {joint_id} for person {person_idx} in {file}: {e}"
                                    )
                                    continue

                                # Optional: Plotting for a specific joint for debugging/visualization
                                # This should be controlled by a config setting or a separate command
                                if (
                                    joint_id == PredJoints.LEFT_ANKLE.value
                                    and person_idx == 0
                                    and config.processing.get(
                                        "enable_filter_plots", False
                                    )
                                ):
                                    plot_dir = os.path.join(root, "plots", filter_name)
                                    os.makedirs(plot_dir, exist_ok=True)
                                    plot_filtering_effect(
                                        original=x_series,
                                        filtered=x_filtered,
                                        title=f"X - {subject} {action} Joint {joint_id} ({filter_name})",
                                        save_path=os.path.join(
                                            plot_dir, f"x_{joint_id}_{filter_name}.png"
                                        ),
                                    )
                                    plot_filtering_effect(
                                        original=y_series,
                                        filtered=y_filtered,
                                        title=f"Y - {subject} {action} Joint {joint_id} ({filter_name})",
                                        save_path=os.path.join(
                                            plot_dir, f"y_{joint_id}_{filter_name}.png"
                                        ),
                                    )

                    # Save the filtered results
                    output_folder_for_video = os.path.join(
                        root, filter_name
                    )  # Save alongside original JSON
                    save_filtered_keypoints(
                        output_folder_for_video,
                        json_path,
                        filtered_pred_keypoints,
                        filter_name,
                    )

                except Exception as e:
                    logger.error(f"Failed to process {json_path} for filtering: {e}")

    logger.info("Keypoint filtering pipeline finished.")

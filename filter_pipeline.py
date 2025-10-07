import os
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from itertools import product
from pathlib import Path
from copy import deepcopy
from config.pipeline_config import PipelineConfig
from config.global_config import GlobalConfig
from utils.import_utils import import_class_from_string
from utils.plot import plot_filtering_effect
from utils.standard_saver import save_standard_format
from filtering_and_data_cleaning.filter_registry import FILTER_FN_MAP
from filtering_and_data_cleaning.preprocessing_utils import TimeSeriesPreprocessor

logger = logging.getLogger(__name__)


class KeypointFilterProcessor:
    def __init__(
        self, config: PipelineConfig, filter_name: str, filter_kwargs: Dict[str, Any]
    ):
        self.config = config
        self.filter_name = filter_name
        self.filter_kwargs = filter_kwargs
        self.input_dir = self.config.filter.input_dir
        self.pred_enum = import_class_from_string(config.dataset.keypoint_format)
        self.custom_output_dir = None  # Will be set later

        self.enable_outlier_removal = getattr(config.filter.outlier_removal, "enable")
        self.outlier_method = getattr(config.filter.outlier_removal, "method")
        self.outlier_params = getattr(config.filter.outlier_removal, "params", {})

        self.enable_interp = getattr(config.filter, "enable_interpolation")
        self.interpolation_kind = getattr(config.filter, "interpolation_kind")
        self.joints_to_filter = self._get_joints_to_filter()
        self.filter_fn = self._get_filter_function()

    def _get_joints_to_filter(self) -> List[int]:
        configured_joints = getattr(self.config.filter, "joints_to_filter", [])
        if configured_joints:
            try:
                joint_indices = [
                    self.pred_enum[j].value
                    for j in configured_joints
                    if j in self.pred_enum.__members__
                ]
                if not joint_indices:
                    logger.warning(
                        "No valid joints found in joints_to_filter configuration"
                    )
                return joint_indices
            except Exception as e:
                logger.warning(f"Error parsing joints_to_filter from config: {e}")
                return []
            return []

    def _get_filter_function(self):
        if self.filter_name not in FILTER_FN_MAP:
            raise ValueError(
                f"Unknown filter: {self.filter_name}. Available: {list(FILTER_FN_MAP.keys())}"
            )
        return FILTER_FN_MAP[self.filter_name]

    def process_directory(self):
        for root, _, files in os.walk(self.input_dir):
            for file in files:
                if file.endswith(".json"):
                    json_path = os.path.join(root, file)
                    self._process_file(json_path, root)
        logger.info("Keypoint filtering pipeline finished.")

    def _process_file(self, json_path: str, root: str):
        logger.info(f"Applying filter to: {json_path}")
        try:
            # Load the original JSON file
            with open(json_path, "r") as f:
                pred_data = json.load(f)

            # Extract persons data and detection config
            if "persons" not in pred_data or not pred_data["persons"]:
                logger.warning(f"No persons found in {json_path}")
                return

            self.original_detection_config = pred_data.get("detection_config", {})

            # Process each person independently to maintain proper structure
            filtered_variants = self._apply_filter_to_persons(
                pred_data["persons"], root
            )

            # Convert json_path to Path
            json_path_obj = Path(json_path)

            # Find the anchor index (e.g., "S1", "S2", etc.)
            try:
                anchor_index = next(
                    i
                    for i, part in enumerate(json_path_obj.parts)
                    if part.startswith("S")
                )
                # Construct relative path from anchor up to parent of .json file
                relative_subdir = str(Path(*json_path_obj.parts[anchor_index:-1]))
            except StopIteration:
                logger.warning(
                    f"Could not find anchor starting with 'S' in path: {json_path}"
                )
                relative_subdir = None

            # Save each param variant result using StandardDataSaver
            for suffix, filtered_persons, filter_params in filtered_variants:
                # Ensure output directory is set
                if not self.custom_output_dir:
                    logger.error("Output directory not set. Cannot save results.")
                    return

                # Prepare output directory structure
                output_folder = os.path.join(
                    self.custom_output_dir,
                    f"{self.filter_name}_{suffix}",
                )

                # Reconstruct the original pred_data structure with filtered persons
                filtered_pred_data = deepcopy(pred_data)
                filtered_pred_data["persons"] = filtered_persons

                # Save using the standard saver with processing metadata
                saved_paths = save_standard_format(
                    data=filtered_pred_data,
                    output_dir=output_folder,
                    original_file_path=json_path,
                    suffix="",  # suffix already included in folder name
                    relative_subdir=relative_subdir,
                    save_json=True,
                    save_pickle=True,
                    save_video_overlay=True,  # Enable video overlay
                    video_input_dir=root,  # Search for video in the same directory
                    video_name=None,  # Will be extracted from data
                    processing_metadata={
                        "filter_name": self.filter_name,
                        "filter_parameters": filter_params,
                        "original_file": json_path,
                        "pipeline": "filtering",
                    },
                )

                logger.info(f"Filter variant '{suffix}' saved. Files: {saved_paths}")

        except Exception as e:
            logger.error(f"Failed to process {json_path}: {e}")

    def _expand_filter_params(self) -> List[Dict[str, Any]]:
        def parse_value(val):
            if isinstance(val, str) and val.strip().startswith("range("):
                try:
                    return list(eval(val.strip()))
                except Exception as e:
                    logger.warning(f"Could not parse range expression '{val}': {e}")
                    return [val]
            elif isinstance(val, list):
                return val
            else:
                return [val]

        keys = list(self.filter_kwargs.keys())
        values = [parse_value(self.filter_kwargs[k]) for k in keys]
        return [dict(zip(keys, v)) for v in product(*values)]

    def _apply_filter_to_persons(
        self, persons_data: List[Dict], root: str
    ) -> List[Tuple[str, List[Dict], Dict[str, Any]]]:
        """Apply filtering to each person's poses independently."""
        param_variants = self._expand_filter_params()
        results = []

        for param_set in param_variants:
            label_suffix = "_".join(f"{k}{v}" for k, v in param_set.items())
            filtered_persons = []

            for person in persons_data:
                if "poses" not in person or not person["poses"]:
                    # Keep person structure even if no poses
                    filtered_persons.append(deepcopy(person))
                    continue

                person_poses = person["poses"]
                filtered_person = deepcopy(person)

                # Apply filtering to this person's poses
                filtered_poses = self._apply_filter_to_person_poses(
                    person_poses, param_set, root
                )
                filtered_person["poses"] = filtered_poses
                filtered_persons.append(filtered_person)

            results.append((label_suffix, filtered_persons, param_set))

        return results

    def _apply_filter_to_person_poses(
        self, poses: List[Dict], param_set: Dict[str, Any], root: str
    ) -> List[Dict]:
        """Apply filtering to a single person's poses."""
        if not poses:
            return poses

        filtered_poses = deepcopy(poses)

        # Get joints to filter
        if not self.joints_to_filter:
            try:
                # Get total joints from first pose's keypoints
                total_joints = len(poses[0]["keypoints"])
                joints_to_filter = list(range(total_joints))
                logger.info(
                    f"No joints_to_filter specified — applying filter to all {total_joints} joints."
                )
            except Exception:
                logger.warning("Unable to determine total joints, skipping filtering.")
                return poses
        else:
            joints_to_filter = self.joints_to_filter

        # Apply filtering to each joint
        for joint_id in joints_to_filter:
            x_series, y_series = [], []

            # Extract time series for this joint
            for pose in poses:
                try:
                    kp = pose["keypoints"][joint_id]  # [x, y] or [x, y, score]
                    x_series.append(kp[0])
                    y_series.append(kp[1])
                except Exception:
                    x_series.append(np.nan)
                    y_series.append(np.nan)

            x_series = np.array(x_series, dtype=np.float64)
            y_series = np.array(y_series, dtype=np.float64)

            if np.all(np.isnan(x_series)) or np.all(np.isnan(y_series)):
                continue

            try:
                # Filtering pipeline
                x_proc, y_proc = x_series.copy(), y_series.copy()

                if self.enable_outlier_removal or self.enable_interp:
                    preprocessor = TimeSeriesPreprocessor(
                        method=self.outlier_method
                        if self.enable_outlier_removal
                        else None,
                        interpolation=self.interpolation_kind
                        if self.enable_interp
                        else None,
                    )
                    try:
                        x_proc = preprocessor.clean(
                            x_proc,
                            **(
                                self.outlier_params
                                if self.enable_outlier_removal
                                else {}
                            ),
                        )
                        y_proc = preprocessor.clean(
                            y_proc,
                            **(
                                self.outlier_params
                                if self.enable_outlier_removal
                                else {}
                            ),
                        )
                    except Exception as e:
                        logger.warning(
                            f"Preprocessing failed on joint {joint_id}. Proceeding with raw series. Error: {e}"
                        )

                # Apply filtering
                x_filt = self.filter_fn(x_proc, **param_set)
                y_filt = self.filter_fn(y_proc, **param_set)

                # Update filtered values in poses
                for i, pose in enumerate(filtered_poses):
                    try:
                        pose["keypoints"][joint_id][0] = float(x_filt[i])
                        pose["keypoints"][joint_id][1] = float(y_filt[i])
                    except (IndexError, TypeError):
                        logger.warning(
                            f"Could not update keypoint {joint_id} for pose {i}"
                        )

                # Optional plotting (only for first person's left ankle)
                if joint_id == self.pred_enum.LEFT_ANKLE.value and getattr(
                    self.config.filter, "enable_filter_plots", False
                ):
                    label_suffix = "_".join(f"{k}{v}" for k, v in param_set.items())
                    plot_dir = os.path.join(
                        root, "plots", f"{self.filter_name}_{label_suffix}"
                    )
                    os.makedirs(plot_dir, exist_ok=True)
                    plot_filtering_effect(
                        x_series,
                        x_filt,
                        title=f"X - Joint {joint_id} ({self.filter_name})",
                        save_path=os.path.join(plot_dir, f"x_{joint_id}.png"),
                    )
                    plot_filtering_effect(
                        y_series,
                        y_filt,
                        title=f"Y - Joint {joint_id} ({self.filter_name})",
                        save_path=os.path.join(plot_dir, f"y_{joint_id}.png"),
                    )

            except Exception as e:
                logger.warning(f"Filter error on joint {joint_id}: {e}")

        return filtered_poses

    def _apply_filter_to_data(
        self, keypoints_frames: List[Dict], root: str
    ) -> List[Tuple[str, List[Dict], Dict[str, Any]]]:
        param_variants = self._expand_filter_params()
        results = []

        for param_set in param_variants:
            frames = deepcopy(keypoints_frames)

            # Get number of persons from first frame's keypoints
            first_frame = frames[0]
            if "keypoints" in first_frame and first_frame["keypoints"]:
                num_persons = len(first_frame["keypoints"])
            else:
                num_persons = 0

            label_suffix = "_".join(f"{k}{v}" for k, v in param_set.items())

            # If no joints specified, use all available joints
            if not self.joints_to_filter:
                try:
                    # Get total joints from first person's keypoints
                    total_joints = len(frames[0]["keypoints"][0])
                    joints_to_filter = list(range(total_joints))
                    logger.info(
                        f"No joints_to_filter specified — applying filter to all {total_joints} joints."
                    )
                except Exception:
                    logger.warning(
                        "Unable to determine total joints, skipping filtering."
                    )
                    continue
            else:
                joints_to_filter = self.joints_to_filter

            # Filtering process - Apply to keypoints
            for person_idx in range(num_persons):
                for joint_id in joints_to_filter:
                    x_series, y_series = [], []

                    for frame in frames:
                        if person_idx >= len(frame["keypoints"]):
                            x_series.append(np.nan)
                            y_series.append(np.nan)
                            continue

                        try:
                            # Access keypoints directly from frame
                            kp = frame["keypoints"][person_idx][
                                joint_id
                            ]  # [x, y] or [x, y, score]
                            x_series.append(kp[0])
                            y_series.append(kp[1])
                        except Exception:
                            x_series.append(np.nan)
                            y_series.append(np.nan)

                    x_series = np.array(x_series, dtype=np.float64)
                    y_series = np.array(y_series, dtype=np.float64)

                    if np.all(np.isnan(x_series)) or np.all(np.isnan(y_series)):
                        continue

                    try:
                        # Filtering pipeline
                        x_proc, y_proc = x_series.copy(), y_series.copy()

                        if self.enable_outlier_removal or self.enable_interp:
                            preprocessor = TimeSeriesPreprocessor(
                                method=self.outlier_method
                                if self.enable_outlier_removal
                                else None,
                                interpolation=self.interpolation_kind
                                if self.enable_interp
                                else None,
                            )
                            try:
                                x_proc = preprocessor.clean(
                                    x_proc,
                                    **(
                                        self.outlier_params
                                        if self.enable_outlier_removal
                                        else {}
                                    ),
                                )
                                y_proc = preprocessor.clean(
                                    y_proc,
                                    **(
                                        self.outlier_params
                                        if self.enable_outlier_removal
                                        else {}
                                    ),
                                )
                            except Exception as e:
                                logger.warning(
                                    f"Preprocessing failed on joint {joint_id}, person {person_idx}. Proceeding with raw series. Error: {e}"
                                )

                        # Apply filtering
                        x_filt = self.filter_fn(x_proc, **param_set)
                        y_filt = self.filter_fn(y_proc, **param_set)

                        # Update filtered values in keypoints
                        for i, frame in enumerate(frames):
                            if person_idx < len(frame["keypoints"]):
                                frame["keypoints"][person_idx][joint_id][0] = float(
                                    x_filt[i]
                                )
                                frame["keypoints"][person_idx][joint_id][1] = float(
                                    y_filt[i]
                                )

                        # Optional plotting
                        if (
                            joint_id == self.pred_enum.LEFT_ANKLE.value
                            and person_idx == 0
                            and getattr(
                                self.config.filter, "enable_filter_plots", False
                            )
                        ):
                            plot_dir = os.path.join(
                                root, "plots", f"{self.filter_name}_{label_suffix}"
                            )
                            os.makedirs(plot_dir, exist_ok=True)
                            plot_filtering_effect(
                                x_series,
                                x_filt,
                                title=f"X - Joint {joint_id} ({self.filter_name})",
                                save_path=os.path.join(plot_dir, f"x_{joint_id}.png"),
                            )
                            plot_filtering_effect(
                                y_series,
                                y_filt,
                                title=f"Y - Joint {joint_id} ({self.filter_name})",
                                save_path=os.path.join(plot_dir, f"y_{joint_id}.png"),
                            )

                    except Exception as e:
                        logger.warning(
                            f"Filter error on joint {joint_id}, person {person_idx}: {e}"
                        )

            # Just return the filtered frames with a label - no formatting here
            results.append((label_suffix, frames, param_set))

        return results


def run_keypoint_filtering_from_config(
    pipeline_config: PipelineConfig,
    global_config: GlobalConfig,
    output_dir: Optional[str] = None,
):
    filter_name = pipeline_config.filter.name
    filter_kwargs = pipeline_config.filter.params or {}

    processor = KeypointFilterProcessor(
        config=pipeline_config, filter_name=filter_name, filter_kwargs=filter_kwargs
    )

    processor.config.paths.input_dir = pipeline_config.filter.input_dir

    # Set output if passed
    if output_dir:
        processor.custom_output_dir = output_dir

    processor.process_directory()

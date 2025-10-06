import os
import json
import logging
import pickle
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from itertools import product
from pathlib import Path
import cv2

from config.pipeline_config import PipelineConfig
from config.global_config import GlobalConfig
from utils.import_utils import import_class_from_string
from utils.plot import plot_filtering_effect
from filtering_and_data_cleaning.filter_registry import FILTER_FN_MAP
from filtering_and_data_cleaning.preprocessing_utils import TimeSeriesPreprocessor
from utils.video_format_utils import get_video_format_info

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
        else:
            logger.warning(
                "No joints_to_filter specified in configuration - no joints will be processed"
            )
            return []

    def _get_filter_function(self):
        if self.filter_name not in FILTER_FN_MAP:
            raise ValueError(
                f"Unknown filter: {self.filter_name}. Available: {list(FILTER_FN_MAP.keys())}"
            )
        return FILTER_FN_MAP[self.filter_name]

    def _overlay_keypoints_on_video(
        self, video_path: str, keypoints_frames: List[Dict], output_path: str
    ):
        """
        Overlay keypoints on video frames and save new video.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video {video_path}")
            return

        # Video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Get fourcc and extension from input video
        fourcc, input_extension = get_video_format_info(video_path)

        # Update output path to use same extension
        output_path_with_ext = Path(output_path).with_suffix(input_extension)
        out = cv2.VideoWriter(str(output_path_with_ext), fourcc, fps, (width, height))

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret or frame_idx >= len(keypoints_frames):
                break

            frame_data = keypoints_frames[frame_idx]["keypoints"]

            for person in frame_data:
                joints = person["keypoints"][0]
                for joint in joints:
                    x, y = int(joint[0]), int(joint[1])
                    if not np.isnan(x) and not np.isnan(y):
                        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            out.write(frame)
            frame_idx += 1

        cap.release()
        out.release()
        logger.info(f"Filtered video saved at: {output_path}")

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

            # Extract frames and detection config
            frames = []
            if "persons" in pred_data:
                for person in pred_data["persons"]:
                    if "poses" in person:
                        frames.extend(person["poses"])
            if not frames:
                logger.warning(f"No keypoints found in {json_path}")
                return

            self.original_detection_config = pred_data.get("detection_config", {})

            # Apply filtering to frames only
            filtered_variants = self._apply_filter_to_data(frames, root)

            # Convert json_path to Path
            json_path_obj = Path(json_path)

            # Find the anchor index (e.g., "S1", "S2", etc.)
            anchor_index = next(
                i for i, part in enumerate(json_path_obj.parts) if part.startswith("S")
            )

            # Construct relative path from anchor up to parent of .json file
            # excludes the filename
            relative_subdir = Path(*json_path_obj.parts[anchor_index:-1])

            # Save each param variant result
            for suffix, filtered_frames in filtered_variants:
                filtered_keypoints = {
                    "keypoints": filtered_frames,
                    "detection_config": self.original_detection_config,
                }

                output_folder = os.path.join(
                    self.custom_output_dir,
                    f"{self.filter_name}_{suffix}",
                    relative_subdir,
                )
                self._save_filtered(json_path, filtered_keypoints, output_folder)
                self._save_as_pickle(json_path, filtered_keypoints, output_folder)
            video_name = json_path.replace(".json", ".avi")
            if os.path.exists(video_name):
                video_output_path = os.path.join(
                    output_folder, os.path.basename(video_name)
                )
                self._overlay_keypoints_on_video(
                    video_name, filtered_frames, video_output_path
                )
            else:
                logger.warning(f"Video file not found for overlay: {video_name}")

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

    def _apply_filter_to_data(
        self, keypoints_frames: List[Dict], root: str
    ) -> List[Tuple[str, List[Dict]]]:
        param_variants = self._expand_filter_params()
        results = []

        for param_set in param_variants:
            frames = json.loads(json.dumps(keypoints_frames))  # deep copy
            num_persons = len(frames[0]["keypoints"]) if "keypoints" in frames[0] else 0
            label_suffix = "_".join(f"{k}{v}" for k, v in param_set.items())

            if not self.joints_to_filter:
                try:
                    total_joints = len(frames[0]["keypoints"][0]["keypoints"][0])
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

            for person_idx in range(num_persons):
                for joint_id in joints_to_filter:
                    x_series, y_series = [], []

                    for frame in frames:
                        if person_idx >= len(frame["keypoints"]):
                            x_series.append(np.nan)
                            y_series.append(np.nan)
                            continue

                        try:
                            kp = frame["keypoints"][person_idx]["keypoints"][0][
                                joint_id
                            ]
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
                        # Step 1: Start with original series
                        x_proc, y_proc = x_series.copy(), y_series.copy()

                        # Step 2: Preprocess if needed
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

                        x_filt = self.filter_fn(x_proc, **param_set)
                        y_filt = self.filter_fn(y_proc, **param_set)

                        for i, frame in enumerate(frames):
                            frame["keypoints"][person_idx]["keypoints"][0][joint_id][
                                0
                            ] = float(x_filt[i])
                            frame["keypoints"][person_idx]["keypoints"][0][joint_id][
                                1
                            ] = float(y_filt[i])

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

            results.append((label_suffix, frames))

        return results  # list of (suffix, filtered_frames)

    def _save_filtered(self, original_path: str, data: List[Dict], output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        # keep original filename
        out_path = os.path.join(output_dir, os.path.basename(original_path))
        with open(out_path, "w") as f:
            json.dump(data, f, indent=4)
        logger.info(f"Filtered keypoints saved to: {out_path}")

    def _save_as_pickle(self, original_path: str, data: List[Dict], output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        pkl_path = os.path.join(
            output_dir, os.path.basename(original_path).replace(".json", ".pkl")
        )
        with open(pkl_path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"Filtered keypoints (pickle) saved to: {pkl_path}")


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

import os
import json
import logging
import pickle
from typing import Dict, Any, List
import numpy as np

from config.base import GlobalConfig
from utils.video_info import extract_video_info
from utils.joint_enum import PredJoints
from utils.plotting import plot_filtering_effect
from post_processing.filter_library import FILTER_FN_MAP

logger = logging.getLogger(__name__)


class KeypointFilterProcessor:
    def __init__(
        self, config: GlobalConfig, filter_name: str, filter_kwargs: Dict[str, Any]
    ):
        self.config = config
        self.filter_name = filter_name
        self.filter_kwargs = filter_kwargs

        self.enable_iqr = config.processing.get("enable_iqr", False)
        self.enable_interp = config.processing.get("enable_interpolation", True)
        self.iqr_multiplier = config.processing.get("iqr_multiplier", 1.5)
        self.interpolation_kind = config.processing.get("interpolation_kind", "linear")
        self.joints_to_filter = self._get_joints_to_filter()
        self.filter_fn = self._get_filter_function()

    def _get_joints_to_filter(self) -> List[int]:
        configured_joints = self.config.processing.get("joints_to_filter", [])
        if configured_joints:
            try:
                return [
                    PredJoints[j].value
                    for j in configured_joints
                    if j in PredJoints.__members__
                ]
            except Exception as e:
                logger.warning(f"Error parsing joints_to_filter from config: {e}")
        return [
            PredJoints.LEFT_ANKLE.value,
            PredJoints.RIGHT_ANKLE.value,
            PredJoints.LEFT_HIP.value,
            PredJoints.RIGHT_HIP.value,
            PredJoints.LEFT_KNEE.value,
            PredJoints.RIGHT_KNEE.value,
        ]

    def _get_filter_function(self):
        if self.filter_name not in FILTER_FN_MAP:
            raise ValueError(
                f"Unknown filter: {self.filter_name}. Available: {list(FILTER_FN_MAP.keys())}"
            )
        return FILTER_FN_MAP[self.filter_name]

    def process_directory(self):
        input_dir = self.config.paths.output_dir

        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.endswith(".json") and not file.endswith("_filtered.json"):
                    json_path = os.path.join(root, file)
                    self._process_file(json_path, root)

        logger.info("Keypoint filtering pipeline finished.")

    def _process_file(self, json_path: str, root: str):
        video_info = extract_video_info(os.path.basename(json_path), root)
        if not video_info:
            logger.warning(f"Could not extract video info from {json_path}. Skipping.")
            return

        subject, action, camera_idx = video_info
        action_group = action.replace(" ", "_")
        logger.info(f"Applying filter to {json_path}")

        try:
            with open(json_path, "r") as f:
                pred_keypoints = json.load(f)
            filtered_keypoints = self._apply_filter_to_data(
                pred_keypoints, subject, action, root
            )

            output_folder = os.path.join(root, self.filter_name)
            self._save_filtered(json_path, filtered_keypoints, output_folder)
            self._save_as_pickle(json_path, filtered_keypoints, output_folder)

        except Exception as e:
            logger.error(f"Failed to process {json_path}: {e}")

    def _apply_filter_to_data(
        self, keypoints_data, subject, action, root
    ) -> List[Dict]:
        data = json.loads(json.dumps(keypoints_data))

        for frame_idx, frame_data in enumerate(data):
            for person_idx, person_data in enumerate(frame_data.get("keypoints", [])):
                kp_series = {jid: {"x": [], "y": []} for jid in self.joints_to_filter}

                for joint_id in self.joints_to_filter:
                    kp = person_data.get("keypoints", {}).get(joint_id)
                    if kp:
                        kp_series[joint_id]["x"].append(kp[0])
                        kp_series[joint_id]["y"].append(kp[1])
                    else:
                        kp_series[joint_id]["x"].append(float("nan"))
                        kp_series[joint_id]["y"].append(float("nan"))

                for joint_id in self.joints_to_filter:
                    x_series = np.array(kp_series[joint_id]["x"], dtype=np.float64)
                    y_series = np.array(kp_series[joint_id]["y"], dtype=np.float64)

                    if np.all(np.isnan(x_series)) or np.all(np.isnan(y_series)):
                        continue

                    try:
                        x_proc, y_proc = x_series, y_series

                        if self.enable_iqr:
                            from post_processing.preprocessing_utils import (
                                remove_outliers_iqr,
                            )

                            x_proc = remove_outliers_iqr(x_proc, self.iqr_multiplier)
                            y_proc = remove_outliers_iqr(y_proc, self.iqr_multiplier)

                        if self.enable_interp:
                            from post_processing.preprocessing_utils import (
                                interpolate_missing_values,
                            )

                            x_proc = interpolate_missing_values(
                                x_proc, kind=self.interpolation_kind
                            )
                            y_proc = interpolate_missing_values(
                                y_proc, kind=self.interpolation_kind
                            )

                        x_filt = self.filter_fn(x_proc, **self.filter_kwargs)
                        y_filt = self.filter_fn(y_proc, **self.filter_kwargs)

                        for i, frame in enumerate(data):
                            if (
                                frame["keypoints"]
                                and joint_id
                                in frame["keypoints"][person_idx]["keypoints"]
                            ):
                                frame["keypoints"][person_idx]["keypoints"][joint_id][
                                    0
                                ] = float(x_filt[i])
                                frame["keypoints"][person_idx]["keypoints"][joint_id][
                                    1
                                ] = float(y_filt[i])

                        if (
                            joint_id == PredJoints.LEFT_ANKLE.value
                            and person_idx == 0
                            and self.config.processing.get("enable_filter_plots", False)
                        ):
                            plot_dir = os.path.join(root, "plots", self.filter_name)
                            os.makedirs(plot_dir, exist_ok=True)
                            plot_filtering_effect(
                                original=x_series,
                                filtered=x_filt,
                                title=f"X - {subject} {action} Joint {joint_id} ({self.filter_name})",
                                save_path=os.path.join(
                                    plot_dir, f"x_{joint_id}_{self.filter_name}.png"
                                ),
                            )
                            plot_filtering_effect(
                                original=y_series,
                                filtered=y_filt,
                                title=f"Y - {subject} {action} Joint {joint_id} ({self.filter_name})",
                                save_path=os.path.join(
                                    plot_dir, f"y_{joint_id}_{self.filter_name}.png"
                                ),
                            )

                    except Exception as e:
                        logger.warning(
                            f"Filter error on joint {joint_id} for person {person_idx}: {e}"
                        )

        return data

    def _save_filtered(self, original_path: str, data: List[Dict], output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(
            output_dir,
            os.path.basename(original_path).replace(
                ".json", f"_{self.filter_name}_filtered.json"
            ),
        )
        with open(out_path, "w") as f:
            json.dump(data, f, indent=4)
        logger.info(f"Filtered keypoints saved to: {out_path}")

    def _save_as_pickle(self, original_path: str, data: List[Dict], output_dir: str):
        pkl_path = os.path.join(
            output_dir,
            os.path.basename(original_path).replace(
                ".json", f"_{self.filter_name}_filtered.pkl"
            ),
        )
        with open(pkl_path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"Filtered keypoints (pickle) saved to: {pkl_path}")

import os
import json
import logging
import pickle
from typing import Dict, Any, List
import numpy as np
from config.global_config import GlobalConfig
from typing import Optional


from config.pipeline_config import PipelineConfig
from utils.video_info import extract_video_info
from utils.joint_enum import PredJoints
from utils.plotting import plot_filtering_effect
from post_processing.filter_registry import FILTER_FN_MAP
from post_processing.preprocessing_utils import TimeSeriesPreprocessor

logger = logging.getLogger(__name__)


class KeypointFilterProcessor:
    def __init__(
        self, config: PipelineConfig, filter_name: str, filter_kwargs: Dict[str, Any]
    ):
        self.config = config
        self.filter_name = filter_name
        self.filter_kwargs = filter_kwargs
        self.input_dir = self.config.filter.input_dir

        self.enable_outlier_removal = getattr(
            config.filter.outlier_removal, "enable", False)
        self.outlier_method = getattr(
            config.filter.outlier_removal, "method", "iqr")
        self.outlier_params = getattr(
            config.filter.outlier_removal, "params", {})

        self.enable_interp = getattr(
            config.filter, "enable_interpolation", True)
        self.interpolation_kind = getattr(
            config.filter, "interpolation_kind", "linear")
        self.joints_to_filter = self._get_joints_to_filter()
        self.filter_fn = self._get_filter_function()

    def _get_joints_to_filter(self) -> List[int]:
        configured_joints = getattr(self.config.filter, "joints_to_filter", [])
        if configured_joints:
            try:
                return [
                    PredJoints[j].value
                    for j in configured_joints
                    if j in PredJoints.__members__
                ]
            except Exception as e:
                logger.warning(
                    f"Error parsing joints_to_filter from config: {e}")
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
        input_dir = self.input_dir
        for root, _, files in os.walk(input_dir):

            for file in files:
                if file.endswith(".json"):
                    json_path = os.path.join(root, file)
                    self._process_file(json_path, root)
        logger.info("Keypoint filtering pipeline finished.")

    def _process_file(self, json_path: str, root: str):
        basename = os.path.splitext(os.path.basename(json_path))[0] + ".avi"
        video_info = extract_video_info(basename, root)

        if not video_info:
            logger.warning(
                f"Could not extract video info from {json_path}. Skipping.")
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
            output_folder = self.custom_output_dir

            self._save_filtered(json_path, filtered_keypoints, output_folder)
            self._save_as_pickle(json_path, filtered_keypoints, output_folder)

        except Exception as e:
            logger.error(f"Failed to process {json_path}: {e}")

    def _apply_filter_to_data(self, keypoints_data, subject, action, root) -> List[Dict]:
        data = json.loads(json.dumps(keypoints_data))

        num_persons = len(data[0]["keypoints"])

        for person_idx in range(num_persons):
            for joint_id in self.joints_to_filter:
                x_series = []
                y_series = []

                for frame in data:
                    if person_idx >= len(frame["keypoints"]):
                        x_series.append(np.nan)
                        y_series.append(np.nan)
                        continue

                    person_data = frame["keypoints"][person_idx]
                    try:
                        kp = person_data["keypoints"][0][joint_id]
                        x_series.append(kp[0])
                        y_series.append(kp[1])
                    except (IndexError, KeyError, TypeError):
                        x_series.append(np.nan)
                        y_series.append(np.nan)

                x_series = np.array(x_series, dtype=np.float64)
                y_series = np.array(y_series, dtype=np.float64)

                if np.all(np.isnan(x_series)) or np.all(np.isnan(y_series)):
                    continue

                try:
                    preprocessor = TimeSeriesPreprocessor(
                        method=self.outlier_method if self.enable_outlier_removal else None,
                        interpolation=self.interpolation_kind if self.enable_interp else None
                    )

                    x_proc = preprocessor.clean(
                        x_series, **self.outlier_params)
                    y_proc = preprocessor.clean(
                        y_series, **self.outlier_params)

                    x_filt = self.filter_fn(x_proc, **self.filter_kwargs)
                    y_filt = self.filter_fn(y_proc, **self.filter_kwargs)

                    for i, frame in enumerate(data):
                        person_kpts = frame["keypoints"][person_idx]["keypoints"]
                        person_kpts[0][joint_id][0] = float(x_filt[i])
                        person_kpts[0][joint_id][1] = float(y_filt[i])

                    if (
                        joint_id == PredJoints.LEFT_ANKLE.value
                        and person_idx == 0
                        and getattr(self.config.filter, "enable_filter_plots", False)
                    ):
                        plot_dir = os.path.join(
                            root, "plots", self.filter_name)
                        os.makedirs(plot_dir, exist_ok=True)
                        plot_filtering_effect(
                            original=x_series,
                            filtered=x_filt,
                            title=f"X - {subject} {action} Joint {joint_id} ({self.filter_name})",
                            save_path=os.path.join(
                                plot_dir, f"x_{joint_id}_{self.filter_name}.png"),
                        )
                        plot_filtering_effect(
                            original=y_series,
                            filtered=y_filt,
                            title=f"Y - {subject} {action} Joint {joint_id} ({self.filter_name})",
                            save_path=os.path.join(
                                plot_dir, f"y_{joint_id}_{self.filter_name}.png"),
                        )

                except Exception as e:
                    logger.warning(
                        f"Filter error on joint {joint_id} for person {person_idx}: {e}")

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


def run_keypoint_filtering_from_config(
    pipeline_config: PipelineConfig,
    global_config: GlobalConfig,
    output_dir: Optional[str] = None
):
    filter_name = pipeline_config.filter.name
    filter_kwargs = pipeline_config.filter.params or {}

    processor = KeypointFilterProcessor(
        config=pipeline_config,
        filter_name=filter_name,
        filter_kwargs=filter_kwargs
    )

    # Use input_dir from pipeline_config.paths.output_dir (e.g., from detect step)
    processor.config.paths.output_dir = pipeline_config.filter.input_dir

    # Use output_dir passed from main_handlers (e.g., run_dir / "filter")
    if output_dir:
        processor.custom_output_dir = output_dir

    processor.process_directory()

import os
import logging
import pandas as pd
import pickle
import numpy as np
from config.pipeline_config import PipelineConfig
from config.global_config import GlobalConfig
from evaluation.evaluation_registry import EVALUATION_METRICS
from utils.video_io import get_video_resolution, rescale_keypoints
from dataset_files.HumanEva.humaneva_metadata import get_humaneva_metadata_from_video
from dataset_files.HumanEva.get_gt_keypoint import GroundTruthLoader
from utils.import_utils import import_class_from_string

logger = logging.getLogger(__name__)


def assess_single_sample(
    subject,
    action,
    camera_idx,
    pred_pkl_path,
    csv_file_path,
    original_video_base,
    pipeline_config: PipelineConfig,  # Added
):
    try:
        cam_name = f"C{camera_idx + 1}"
        safe_action_name = action.replace(" ", "_")

        original_video_path = os.path.join(
            original_video_base,
            subject, "Image_Data",
            f"{safe_action_name}_({cam_name}).avi",
        )

        # Save/load GT from same folder as CSV
        gt_dir = os.path.dirname(csv_file_path)
        gt_pkl_name = f"{subject}_{safe_action_name}_{cam_name}_gt.pkl"
        gt_pkl_path = os.path.join(gt_dir, gt_pkl_name)

        if not os.path.exists(gt_pkl_path):
            logger.info(f"Generating ground truth PKL: {gt_pkl_path}")
            loader = GroundTruthLoader(csv_file_path)
            keypoints = loader.get_keypoints(
                subject, action, camera_idx, chunk="chunk0")
            with open(gt_pkl_path, "wb") as f:
                pickle.dump({"keypoints": keypoints}, f)

        with open(gt_pkl_path, "rb") as f:
            gt_data = pickle.load(f)
        gt_keypoints = gt_data["keypoints"]  # shape (N, J, 2)

        # Load prediction from .pkl
        with open(pred_pkl_path, "rb") as f:
            pred_data = pickle.load(f)

        pred_keypoints = []
        for frame in pred_data["keypoints"]:
            people = frame["keypoints"]
            if len(people) == 0:
                raise ValueError(f"No people in frame {frame['frame_idx']}")
            keypoints_arr = np.array(people[0]["keypoints"])
            if keypoints_arr.ndim == 3 and keypoints_arr.shape[0] == 1:
                keypoints_arr = keypoints_arr[0]
            pred_keypoints.append(keypoints_arr)  # (J, 2)

        pred_keypoints = np.stack(pred_keypoints, axis=0)

        # Rescale if necessary
        orig_w, orig_h = get_video_resolution(original_video_path)
        pred_video_path = os.path.join(
            os.path.dirname(pred_pkl_path),
            f"{os.path.splitext(os.path.basename(pred_pkl_path))[0]}.avi",
        )

        if os.path.exists(pred_video_path):
            test_w, test_h = get_video_resolution(pred_video_path)
            if (test_w, test_h) != (orig_w, orig_h):
                pred_keypoints = rescale_keypoints(
                    pred_keypoints, orig_w / test_w, orig_h / test_h
                )

        # Get synced start frame from pipeline_config
        try:
            sync_start = pipeline_config.dataset.sync_data["data"][subject][action][camera_idx]
            if sync_start >= len(pred_keypoints):
                logger.warning(
                    f"Sync start index {sync_start} is out of bounds for prediction length {len(pred_keypoints)}")
                return None
        except KeyError:
            logger.warning(
                f"No sync start index for {subject} | {action} | C{camera_idx}")
            sync_start = 0

        # Apply sync trimming
        pred_keypoints = pred_keypoints[sync_start:]
        min_len = min(len(gt_keypoints), len(pred_keypoints))

        return gt_keypoints[:min_len], pred_keypoints[:min_len]

    except Exception as e:
        logger.error(f"Assessment error: {e}")
        return None


class MetricsEvaluator:
    def __init__(self, output_path):
        self.results = []
        self.output_path = output_path

    def evaluate(
        self, calculator, gt, pred, subject, action, camera, metric_name, params
    ):
        if hasattr(calculator, "compute"):
            result = calculator.compute(gt, pred)
        else:
            raise ValueError(f"{metric_name} missing `compute()`.")

        if isinstance(result, tuple) and len(result) == 2:
            joint_names, jointwise_scores = result
            for joint, scores in zip(joint_names, jointwise_scores.T):
                self.results.append(
                    {
                        "subject": subject,
                        "action": action,
                        "camera": camera,
                        "metric": metric_name,
                        "joint": joint,
                        **params,
                        "score": scores.mean(),
                    }
                )
        else:
            self.results.append(
                {
                    "subject": subject,
                    "action": action,
                    "camera": camera,
                    "metric": metric_name,
                    **params,
                    "score": result,
                }
            )

    def save(self):
        df = pd.DataFrame(self.results)
        df.to_excel(self.output_path, index=False)


def run_humaneva_assessment(
    pipeline_config: PipelineConfig,
    global_config: GlobalConfig,
    output_dir: str,
    input_dir: str,
):
    gt_enum_class = import_class_from_string(
        pipeline_config.dataset.joint_enum_module)
    pred_enum_class = import_class_from_string(
        pipeline_config.dataset.keypoint_format)

    logger.info("Running HumanEva assessment...")

    pred_root = (
        pipeline_config.evaluation.input_dir or pipeline_config.detect.output_dir
    )
    csv_file_path = pipeline_config.paths.ground_truth_file
    original_video_base = os.path.join(
        global_config.paths.input_dir, pipeline_config.paths.dataset)

    global_results = []

    for root, _, files in os.walk(pred_root):
        for file in files:
            if not file.endswith(".pkl") or "gt" in file:
                continue

            pred_pkl_path = os.path.join(root, file)
            json_name = file.replace(".pkl", ".json")
            json_path = os.path.join(root, json_name)

            result = get_humaneva_metadata_from_video(json_path)
            if not result:
                logger.warning(f"Could not parse HumanEva info from {file}")
                continue

            subject = result["subject"]
            action = result["action"]
            camera_str = result["camera"]
            camera_idx = int(camera_str[1:]) - 1
            logger.info(f"Evaluating: {subject} | {action} | C{camera_idx}")
            action_group = action.replace(" ", "_")

            sample = assess_single_sample(
                subject,
                action,
                camera_idx,
                pred_pkl_path,
                csv_file_path,
                original_video_base,
                pipeline_config=pipeline_config,
            )
            if not sample:
                continue

            gt, pred = sample
            for metric_cfg in pipeline_config.evaluation.metrics:
                metric_name = metric_cfg["name"]
                params = metric_cfg.get("params", {})

                metric_entry = next(
                    (m for m in EVALUATION_METRICS if m["name"]
                     == metric_name),
                    None,
                )
                if not metric_entry:
                    logger.error(f"Metric '{metric_name}' not found.")
                    continue

                expected = set(metric_entry.get("param_spec", []))
                provided = set(params.keys())
                if expected != provided:
                    raise ValueError(
                        f"Params for '{metric_name}' do not match. Expected {expected}, got {provided}."
                    )

                calculator = metric_entry["class"](
                    **params,
                    gt_enum=gt_enum_class,
                    pred_enum=pred_enum_class,
                )

                evaluator = MetricsEvaluator(output_path=None)
                evaluator.evaluate(
                    calculator,
                    gt,
                    pred,
                    subject,
                    action_group,
                    camera_idx,
                    metric_name,
                    params,
                )
                global_results.extend(evaluator.results)

    if global_results:
        df = pd.DataFrame(global_results)
        parent_folder_name = os.path.basename(
            os.path.dirname(os.path.normpath(pred_root))
        )

        for metric_cfg in pipeline_config.evaluation.metrics:
            metric_name = metric_cfg["name"]
            metric_df = df[df["metric"] == metric_name]
            if not metric_df.empty:
                combined_name = f"{parent_folder_name}_{metric_name}.xlsx"
                combined_path = os.path.join(output_dir, combined_name)
                metric_df.to_excel(combined_path, index=False)
                logger.info(f"Combined result saved: {combined_path}")

    logger.info("HumanEva assessment completed.")

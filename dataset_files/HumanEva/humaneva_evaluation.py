import os
import logging
import pandas as pd
import re
from config.pipeline_config import PipelineConfig
from config.global_config import GlobalConfig
from evaluation.evaluation_registry import EVALUATION_METRICS
from dataset_files.HumanEva.get_gt_keypoint import GroundTruthLoader
from utils.extract_predicted_points import PredictionExtractor
from utils.video_io import get_video_resolution, rescale_keypoints
from dataset_files.HumanEva.humaneva_metadata import get_humaneva_metadata_from_video
from utils.import_utils import import_class_from_string


logger = logging.getLogger(__name__)


import pickle


def assess_single_sample(
    subject,
    action,
    camera_idx,
    json_path,
    pipeline_config,
    global_config,
    csv_file_path,
    original_video_base,
    gt_output_dir=None,  # <--- NEW
):
    try:
        cam_name = f"C{camera_idx + 1}"
        safe_action_name = action.replace(" ", "_")
        original_video_path = os.path.join(
            original_video_base,
            subject,
            "Image_Data",
            f"{safe_action_name}_({cam_name}).avi",
        )

        gt_loader = GroundTruthLoader(csv_file_path)
        gt_keypoints = gt_loader.get_keypoints(
            subject, action, camera_idx, chunk="chunk0"
        )

        # NEW: Save ground truth to pkl
        if gt_output_dir:
            os.makedirs(gt_output_dir, exist_ok=True)
            out_name = f"{subject}_{safe_action_name}_{cam_name}_gt.pkl"
            out_path = os.path.join(gt_output_dir, out_name)
            with open(out_path, "wb") as f:
                pickle.dump(gt_keypoints, f)
            logger.info(f"Saved ground truth to: {out_path}")

        sync_frame_tuple = (
            pipeline_config.dataset.sync_data.get("data", {})
            .get(subject, {})
            .get(action)
        )

        if not sync_frame_tuple:
            logger.warning(f"Missing sync data for {subject}, {action}")
            return None

        sync_frame = sync_frame_tuple[camera_idx]
        frame_range = (sync_frame, sync_frame + len(gt_keypoints))

        pred_loader = PredictionExtractor(json_path, file_format="json")
        pred_keypoints_org = pred_loader.get_keypoint_array(frame_range=frame_range)

        testing_video_path = os.path.join(
            os.path.dirname(json_path),
            f"{os.path.splitext(os.path.basename(json_path))[0]}.avi",
        )

        orig_w, orig_h = get_video_resolution(original_video_path)

        if os.path.exists(testing_video_path):
            testing_w, testing_h = get_video_resolution(testing_video_path)
            if (testing_w, testing_h) != (orig_w, orig_h):
                pred_keypoints = rescale_keypoints(
                    pred_keypoints_org, orig_w / testing_w, orig_h / testing_h
                )
            else:
                pred_keypoints = pred_keypoints_org
        else:
            logger.warning(
                f"No degraded video found at {testing_video_path}, using unscaled predictions."
            )
            pred_keypoints = pred_keypoints_org

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
    gt_enum_class = import_class_from_string(pipeline_config.dataset.joint_enum_module)
    pred_enum_class = import_class_from_string(pipeline_config.dataset.keypoint_format)

    logger.info("Running HumanEva assessment...")

    pred_root = (
        pipeline_config.evaluation.input_dir or pipeline_config.detect.output_dir
    )
    csv_file_path = pipeline_config.paths.ground_truth_file
    original_video_base = input_dir

    # ✅ Create a global evaluator to collect all rows
    global_results = []

    for root, _, files in os.walk(pred_root):
        for file in files:
            if not file.endswith(".json"):
                continue

            json_path = os.path.join(root, file)
            result = get_humaneva_metadata_from_video(json_path)
            if not result:
                logger.warning(f"Could not parse HumanEva info from {json_path}")
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
                json_path,
                pipeline_config,
                global_config,
                csv_file_path,
                original_video_base,
            )
            if not sample:
                continue

            gt, pred = sample
            for metric_cfg in pipeline_config.evaluation.metrics:
                metric_name = metric_cfg["name"]
                params = metric_cfg.get("params", {})

                metric_entry = next(
                    (m for m in EVALUATION_METRICS if m["name"] == metric_name),
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

                # ✅ Local evaluation to gather rows
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

    # ✅ Save combined results at the end
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

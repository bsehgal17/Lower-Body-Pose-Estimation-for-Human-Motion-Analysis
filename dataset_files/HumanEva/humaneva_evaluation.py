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
from dataset_files.HumanEva.humaneva_metadata import get_humaneva_metadata_from_path

logger = logging.getLogger(__name__)


def assess_single_sample(
    subject,
    action,
    camera_idx,
    json_path,
    pipeline_config,
    global_config,
    csv_file_path,
    original_video_base,
):
    try:
        cam_name = f"C{camera_idx + 1}"
        original_video_path = os.path.join(
            original_video_base, subject, "Image_Data", f"{action}_{cam_name}.avi"
        )

        gt_loader = GroundTruthLoader(csv_file_path)
        gt_keypoints = gt_loader.get_keypoints(
            subject, action, camera_idx, chunk="chunk0"
        )

        sync_frame_tuple = pipeline_config.sync_data.data.get(subject, {}).get(action)
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
    pipeline_config: PipelineConfig, global_config: GlobalConfig, output_dir: str
):
    logger.info("Running HumanEva assessment...")

    pred_root = (
        pipeline_config.evaluation.input_dir or pipeline_config.detect.output_dir
    )
    csv_file_path = pipeline_config.paths.ground_truth_file
    original_video_base = global_config.paths.input_dir

    for root, _, files in os.walk(pred_root):
        for file in files:
            if not file.endswith(".json"):
                continue

            json_path = os.path.join(root, file)
            result = get_humaneva_metadata_from_path(json_path)
            if not result:
                logger.warning(f"Could not parse HumanEva info from {json_path}")
                continue

            subject, action, camera_idx = result
            logger.info(f"Evaluating: {subject} | {action} | C{camera_idx + 1}")
            action_group = action.replace(" ", "_")

            excel_name = f"{subject}_{action_group}_C{camera_idx + 1}_assessment.xlsx"
            excel_path = os.path.join(output_dir, excel_name)
            evaluator = MetricsEvaluator(excel_path)

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
                metric_entry = next(
                    (m for m in EVALUATION_METRICS if m["name"] == metric_cfg.name),
                    None,
                )
                if not metric_entry:
                    logger.error(f"Metric '{metric_cfg.name}' not found.")
                    continue

                expected = set(metric_entry.get("param_spec", []))
                provided = set(metric_cfg.params.keys())
                if expected != provided:
                    raise ValueError(
                        f"Params for '{metric_cfg.name}' do not match. Expected {expected}, got {provided}."
                    )

                calculator = metric_entry["class"](**metric_cfg.params)
                evaluator.evaluate(
                    calculator,
                    gt,
                    pred,
                    subject,
                    action_group,
                    camera_idx + 1,
                    metric_cfg.name,
                    metric_cfg.params,
                )

            evaluator.save()
            logger.info(f"Saved: {excel_path}")

    logger.info("HumanEva assessment completed.")

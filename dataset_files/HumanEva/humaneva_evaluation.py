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
    pipeline_config: PipelineConfig,
):
    try:
        cam_name = f"C{camera_idx + 1}"
        safe_action_name = action.replace(" ", "_")

        original_video_path = os.path.join(
            original_video_base,
            subject, "Image_Data",
            f"{safe_action_name}_({cam_name}).avi",
        )

        gt_dir = os.path.dirname(csv_file_path)
        gt_pkl_name = f"{subject}_{safe_action_name}_{cam_name}_gt.pkl"
        gt_pkl_folder = os.path.join(gt_dir, "pickle_files")
        os.makedirs(gt_pkl_folder, exist_ok=True)
        gt_pkl_path = os.path.join(gt_pkl_folder, gt_pkl_name)

        if not os.path.exists(gt_pkl_path):
            logger.info(f"Generating ground truth PKL: {gt_pkl_path}")
            loader = GroundTruthLoader(csv_file_path)
            keypoints = loader.get_keypoints(
                subject, action, camera_idx, chunk="chunk0")
            with open(gt_pkl_path, "wb") as f:
                pickle.dump({"keypoints": keypoints}, f)

        with open(gt_pkl_path, "rb") as f:
            gt_data = pickle.load(f)
        gt_keypoints = gt_data["keypoints"]

        with open(pred_pkl_path, "rb") as f:
            pred_data = pickle.load(f)

        pred_keypoints = []
        for frame in pred_data["keypoints"]:
            people = frame["keypoints"]
            if not people:
                raise ValueError(f"No people in frame {frame['frame_idx']}")
            keypoints_arr = np.array(people[0]["keypoints"])
            if keypoints_arr.ndim == 3 and keypoints_arr.shape[0] == 1:
                keypoints_arr = keypoints_arr[0]
            pred_keypoints.append(keypoints_arr)

        pred_keypoints = np.stack(pred_keypoints, axis=0)

        orig_w, orig_h = get_video_resolution(original_video_path)
        pred_video_path = os.path.join(
            os.path.dirname(pred_pkl_path),
            f"{os.path.splitext(os.path.basename(pred_pkl_path))[0]}.avi",
        )
        if os.path.exists(pred_video_path):
            test_w, test_h = get_video_resolution(pred_video_path)
            if (test_w, test_h) != (orig_w, orig_h):
                pred_keypoints = rescale_keypoints(
                    pred_keypoints, orig_w / test_w, orig_h / test_h)

        try:
            sync_start = pipeline_config.dataset.sync_data["data"][subject][action][camera_idx]
            if sync_start >= len(pred_keypoints):
                logger.warning(
                    f"Sync index {sync_start} exceeds prediction length {len(pred_keypoints)}")
                return None
        except KeyError:
            logger.warning(
                f"No sync index for {subject} | {action} | C{camera_idx}")
            sync_start = 0

        pred_keypoints = pred_keypoints[sync_start:]
        min_len = min(len(gt_keypoints), len(pred_keypoints))
        return gt_keypoints[:min_len], pred_keypoints[:min_len]

    except Exception as e:
        logger.error(f"Assessment error: {e}")
        return None


class MetricsEvaluator:
    def __init__(self, output_path=None):
        self.overall_rows = []
        self.jointwise_rows = []
        self.output_path = output_path
        self.joint_names = None

    def evaluate(self, calculator, gt, pred, subject, action, camera, metric_name, params):
        result = calculator.compute(gt, pred)

        base_row = {
            "subject": subject,
            "action": action,
            "camera": camera
        }

        threshold = params.get('threshold', 0)
        key = f"{metric_name}_{threshold:.2f}"

        if isinstance(result, (float, int, np.floating, np.integer)):
            # Pure overall metric (not derived from joints)
            overall_row = base_row.copy()
            overall_row[f"overall_{key}"] = result
            self.overall_rows.append(overall_row)

        elif isinstance(result, tuple) and len(result) == 2:
            joint_names, jointwise_scores = result
            if self.joint_names is None:
                self.joint_names = joint_names

            # Jointwise metrics (one column per joint)
            jointwise_row = base_row.copy()
            for joint_name, score in zip(joint_names, jointwise_scores):
                jointwise_row[f"{joint_name}_{key}"] = score
            self.jointwise_rows.append(jointwise_row)

    def save(self, output_dir):
        if not (self.overall_rows or self.jointwise_rows):
            return

        parent_folder_name = os.path.basename(
            os.path.dirname(os.path.normpath(self.output_path)))

        # Save overall metrics (only true overall metrics)
        if self.overall_rows:
            df_overall = pd.DataFrame(self.overall_rows)
            df_overall = df_overall.groupby(
                ["subject", "action", "camera"]).first().reset_index()

            overall_path = os.path.join(
                output_dir, f"{parent_folder_name}_overall_metrics.xlsx")
            with pd.ExcelWriter(overall_path, engine='openpyxl') as writer:
                df_overall.to_excel(writer, index=False,
                                    sheet_name='Overall Metrics')

                worksheet = writer.sheets['Overall Metrics']
                for column in worksheet.columns:
                    col_name = column[0].value
                    max_len = max(
                        df_overall[col_name].astype(str).map(len).max(),
                        len(col_name)
                    ) + 2
                    worksheet.column_dimensions[column[0]
                                                .column_letter].width = max_len

        # Save jointwise metrics (detailed per-joint)
        if self.jointwise_rows:
            df_jointwise = pd.DataFrame(self.jointwise_rows)
            df_jointwise = df_jointwise.groupby(
                ["subject", "action", "camera"]).first().reset_index()

            jointwise_path = os.path.join(
                output_dir, f"{parent_folder_name}_jointwise_metrics.xlsx")
            with pd.ExcelWriter(jointwise_path, engine='openpyxl') as writer:
                df_jointwise.to_excel(
                    writer, index=False, sheet_name='Jointwise Metrics')

                worksheet = writer.sheets['Jointwise Metrics']
                for column in worksheet.columns:
                    col_name = column[0].value
                    max_len = max(
                        df_jointwise[col_name].astype(str).map(len).max(),
                        len(col_name)
                    ) + 2
                    worksheet.column_dimensions[column[0]
                                                .column_letter].width = max_len


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

    pred_root = pipeline_config.evaluation.input_dir or pipeline_config.detect.output_dir
    csv_file_path = pipeline_config.paths.ground_truth_file
    original_video_base = os.path.join(
        global_config.paths.input_dir, pipeline_config.paths.dataset)

    evaluator = MetricsEvaluator()

    for root, _, files in os.walk(pred_root):
        for file in files:
            if not file.endswith(".pkl") or "gt" in file:
                continue

            pred_pkl_path = os.path.join(root, file)
            json_path = os.path.join(root, file.replace(".pkl", ".json"))
            result = get_humaneva_metadata_from_video(json_path)
            if not result:
                logger.warning(f"Could not parse metadata from {file}")
                continue

            subject, action, camera_str = result["subject"], result["action"], result["camera"]
            camera_idx = int(camera_str[1:]) - 1
            action_group = action.replace(" ", "_")

            logger.info(f"Evaluating: {subject} | {action} | C{camera_idx}")

            sample = assess_single_sample(
                subject, action, camera_idx, pred_pkl_path,
                csv_file_path, original_video_base, pipeline_config
            )
            if not sample:
                continue

            gt, pred = sample
            for metric_cfg in pipeline_config.evaluation.metrics:
                metric_name = metric_cfg["name"]
                params = metric_cfg.get("params", {})

                metric_entry = next(
                    (m for m in EVALUATION_METRICS if m["name"] == metric_name), None)
                if not metric_entry:
                    logger.error(f"Metric '{metric_name}' not found.")
                    continue

                expected = set(metric_entry.get("param_spec", []))
                provided = set(params.keys())
                if not provided <= expected:
                    raise ValueError(
                        f"Params for '{metric_name}' do not match. "
                        f"Expected subset of {expected}, got {provided}."
                    )

                calculator = metric_entry["class"](
                    params=params, gt_enum=gt_enum_class, pred_enum=pred_enum_class
                )

                evaluator.evaluate(
                    calculator, gt, pred,
                    subject, action_group, camera_idx,
                    metric_name, params
                )

    if evaluator.overall_rows or evaluator.jointwise_rows:
        evaluator.output_path = pred_root
        evaluator.save(output_dir)

    logger.info("HumanEva assessment completed.")

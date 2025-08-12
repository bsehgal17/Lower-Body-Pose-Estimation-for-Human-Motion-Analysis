import os
import logging
import pickle
import numpy as np
from config.pipeline_config import PipelineConfig
from config.global_config import GlobalConfig
from dataset_files.HumanEva.humaneva_metadata import get_humaneva_metadata_from_video
from dataset_files.HumanEva.get_gt_keypoint import GroundTruthLoader
from utils.video_io import get_video_resolution, rescale_keypoints
from utils.import_utils import import_class_from_string
from evaluation.generic_evaluator import MetricsEvaluator, run_assessment

logger = logging.getLogger(__name__)


def humaneva_data_loader(pred_pkl_path, pipeline_config: PipelineConfig, global_config: GlobalConfig):
    """
    Loads and prepares GT and prediction data for a single HumanEva sample.
    Returns (gt_keypoints, pred_keypoints, sample_info) or None.
    """
    try:
        json_path = pred_pkl_path.replace(".pkl", ".json")
        metadata = get_humaneva_metadata_from_video(json_path)
        if not metadata:
            logger.warning(
                f"Could not parse metadata from {os.path.basename(json_path)}")
            return None

        subject, action, camera_str = metadata["subject"], metadata["action"], metadata["camera"]
        camera_idx = int(camera_str[1:]) - 1
        safe_action_name = action.replace(" ", "_")

        # --- GT Loading Logic (HumanEva specific) ---
        original_video_base = os.path.join(
            global_config.paths.input_dir, pipeline_config.paths.dataset
        )
        csv_file_path = pipeline_config.paths.ground_truth_file

        gt_dir = os.path.dirname(csv_file_path)
        gt_pkl_name = f"{subject}_{safe_action_name}_{camera_str}_gt.pkl"
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

        # --- Prediction Loading and Preprocessing ---
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

        # --- Rescaling and Synchronization (HumanEva specific) ---
        original_video_path = os.path.join(
            original_video_base, subject, "Image_Data", f"{safe_action_name}_({camera_str}).avi")
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
                f"No sync index for {subject} | {action} | {camera_str}")
            sync_start = 0

        pred_keypoints = pred_keypoints[sync_start:]
        min_len = min(len(gt_keypoints), len(pred_keypoints))

        sample_info = {
            "subject": subject,
            "action": safe_action_name,
            "camera": camera_idx
        }

        return gt_keypoints[:min_len], pred_keypoints[:min_len], sample_info

    except Exception as e:
        logger.error(f"Assessment error for {pred_pkl_path}: {e}")
        return None


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

    logger.info("Running HumanEva assessment using generic evaluator...")

    pred_root = pipeline_config.evaluation.input_dir or pipeline_config.detect.output_dir
    evaluator = MetricsEvaluator()

    # Define the keys for grouping explicitly here
    grouping_keys = ['subject', 'action', 'camera']

    run_assessment(
        evaluator=evaluator,
        pipeline_config=pipeline_config,
        global_config=global_config,
        input_dir=pred_root,
        output_dir=output_dir,
        gt_enum_class=gt_enum_class,
        pred_enum_class=pred_enum_class,
        data_loader_func=humaneva_data_loader,
        group_keys=grouping_keys  # Pass the explicit keys for grouping
    )

    logger.info("HumanEva assessment completed.")

import os
import logging
import pickle
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from dataclasses import asdict
from huggingface_hub import hf_hub_download

import deeplabcut.pose_estimation_pytorch as dlc_torch
from deeplabcut.utils.make_labeled_video import CreateVideo
from deeplabcut.utils.video_processor import VideoProcessorCV
from utils.video_io import get_video_files
from utils.json_io import save_keypoints_to_json
from config.pipeline_config import PipelineConfig
from config.global_config import GlobalConfig
from pose_estimation.deeplabcut.skeleton_config import bodyparts2connect

logger = logging.getLogger(__name__)


def convert_to_standard_format(predictions):
    """
    Converts predictions into RTMW-style format.
    Filters out invalid detections with all -1s.
    """
    std_formatted = []

    for frame_idx, frame_data in enumerate(predictions):
        kps = np.asarray(frame_data.get("bodyparts", []))
        bbs = np.asarray(frame_data.get("bboxes", []))

        frame_entry = {
            "frame_idx": frame_idx,
            "keypoints": []
        }

        if kps.ndim == 2:
            kps = kps[None, ...]  # Convert to shape (1, J, 3)

        for i in range(kps.shape[0]):
            kp = kps[i]
            bbox = bbs[i] if i < len(bbs) else [-1, -1, -1, -1]

            # Skip if all keypoints are [-1, -1, -1]
            if np.all(kp == -1):
                continue

            # Skip if bbox is exactly [-1, -1, -1, -1]
            if np.all(bbox == -1):
                continue

            frame_entry["keypoints"].append({
                "keypoints": kp[:, :2].tolist(),
                "scores": kp[:, 2].tolist(),
                "bboxes": bbox.tolist()
            })

        std_formatted.append(frame_entry)

    return std_formatted


def run_detection_pipeline(pipeline_config: PipelineConfig, global_config: GlobalConfig, input_dir: str, output_dir: str):
    logger.info("Initializing DeepLabCut detection pipeline...")

    from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights

    device = pipeline_config.processing.device
    max_detections = 30
    detection_thresh = pipeline_config.processing.detection_threshold or 0.3

    weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
    detector = fasterrcnn_mobilenet_v3_large_fpn(
        weights=weights, box_score_thresh=detection_thresh).to(device).eval()
    preprocess = weights.transforms()

    model_dir = Path("hf_files")
    model_dir.mkdir(exist_ok=True)

    pose_cfg = hf_hub_download(
        repo_id="DeepLabCut/HumanBody",
        filename="rtmpose-x_simcc-body7_pytorch_config.yaml",
        local_dir=model_dir
    )
    pt_path = hf_hub_download(
        repo_id="DeepLabCut/HumanBody",
        filename="rtmpose-x_simcc-body7.pt",
        local_dir=model_dir
    )
    pose_cfg = dlc_torch.config.read_config_as_dict(pose_cfg)

    runner = dlc_torch.get_pose_inference_runner(
        pose_cfg,
        snapshot_path=pt_path,
        batch_size=16,
        max_individuals=max_detections,
    )

    video_files = get_video_files(input_dir, global_config.video.extensions)
    if not video_files:
        logger.warning(f"No video files found in {input_dir}")
        return

    for video_path in video_files:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        rel_path = os.path.relpath(video_path, input_dir)
        current_save_dir = os.path.join(
            output_dir, os.path.dirname(rel_path), video_name)
        os.makedirs(current_save_dir, exist_ok=True)

        output_json_file = os.path.join(current_save_dir, f"{video_name}.json")
        output_pkl_file = os.path.join(current_save_dir, f"{video_name}.pkl")
        overlay_video_file = os.path.join(
            current_save_dir, f"{video_name}_overlay.mp4")

        logger.info(f"Processing: {video_path}")

        # Step 1: Run Detection
        video = dlc_torch.VideoIterator(video_path)
        context = []

        with torch.no_grad():
            for frame in tqdm(video, desc="Running object detection"):
                batch = [preprocess(Image.fromarray(frame)).to(device)]
                predictions = detector(batch)[0]
                bboxes = predictions["boxes"].detach().cpu().numpy()
                labels = predictions["labels"].detach().cpu().numpy()

                human_bboxes = [bbox for bbox, label in zip(
                    bboxes, labels) if label == 1]
                if human_bboxes:
                    bboxes = np.stack(human_bboxes)
                    bboxes[:, 2] -= bboxes[:, 0]
                    bboxes[:, 3] -= bboxes[:, 1]
                    bboxes = bboxes[:max_detections]
                else:
                    bboxes = np.zeros((0, 4))

                context.append({"bboxes": bboxes})

        video.set_context(context)

        # Step 2: Run Pose Estimation
        video.reset()
        predictions = runner.inference(
            tqdm(video, desc="Running pose estimation"))

        # Save JSON/PKL
        keypoints_rtmw = convert_to_standard_format(predictions)
        detector_config_dict = asdict(pipeline_config.processing)
        save_keypoints_to_json(keypoints_rtmw, current_save_dir,
                               video_name, detector_config=detector_config_dict)

        with open(output_pkl_file, "wb") as f:
            pickle.dump({"keypoints": keypoints_rtmw,
                        "detection_config": detector_config_dict}, f)

        logger.info(
            f"Saved keypoints to: {output_json_file} and {output_pkl_file}")
        wrapped_preds = {}
        for idx, frame in enumerate(predictions):
            if "bodyparts" in frame and isinstance(frame["bodyparts"], np.ndarray):
                # Use only first individual
                bp = frame["bodyparts"]
                if bp.ndim == 3 and bp.shape[0] > 0:
                    # only first person
                    wrapped_preds[idx] = {"bodyparts": bp[0:1]}

        # Step 3: Overlay video
        try:
            df = dlc_torch.build_predictions_dataframe(
                scorer="rtmpose-body7",
                predictions=wrapped_preds,
                parameters=dlc_torch.PoseDatasetParameters(
                    bodyparts=pose_cfg["metadata"]["bodyparts"],
                    unique_bpts=pose_cfg["metadata"]["unique_bodyparts"],
                    individuals=["single"]
                )
            )

            clip = VideoProcessorCV(
                str(video_path), sname=overlay_video_file, codec="mp4v")

            CreateVideo(
                clip,
                df,
                pcutoff=0.4,
                dotsize=3,
                colormap="rainbow",
                bodyparts2plot=pose_cfg["metadata"]["bodyparts"],
                trailpoints=0,
                cropping=False,
                x1=0, x2=clip.w,
                y1=0, y2=clip.h,
                bodyparts2connect=bodyparts2connect,
                skeleton_color="w",
                draw_skeleton=True,
                displaycropped=True,
                color_by="bodypart",
            )

            logger.info(f"Overlay video saved to: {overlay_video_file}")

        except Exception as e:
            logger.exception(f"Failed to create overlay video: {e}")

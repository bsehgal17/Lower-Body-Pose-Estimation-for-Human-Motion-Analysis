import os
import logging
import numpy as np
from tqdm import tqdm
import cv2
from dataclasses import asdict
from mmpose.apis import inference_topdown, init_model as init_pose_model
from mmdet.apis import inference_detector, init_detector
from mmpose.utils import adapt_mmdet_pipeline

from utils.video_io import get_video_files
from utils.standard_saver import save_standard_format, SavedData
from utils.data_structures import VideoData
from config.pipeline_config import PipelineConfig
from config.global_config import GlobalConfig
from pose_estimation.DWPose.skeleton_config import bodyparts2connect

logger = logging.getLogger(__name__)


def convert_to_standard_format(predictions):
    """
    Updated version to handle both 2D and 3D keypoints
    """
    std_formatted = []

    for frame_idx, frame_data in enumerate(predictions):
        frame_entry = {"frame_idx": frame_idx, "keypoints": []}

        for person_kpts in frame_data:
            # Handle both (x,y) and (x,y,score) formats
            if person_kpts.shape[1] == 2:  # Only x,y coordinates
                scores = np.ones(len(person_kpts))  # Default score=1
                person_kpts = np.column_stack([person_kpts, scores])
            elif person_kpts.shape[1] != 3:
                continue  # Skip invalid formats

            # Skip if all keypoints are invalid
            if np.all(person_kpts[:, 2] <= 0):
                continue

            # Calculate bounding box
            valid_mask = person_kpts[:, 2] > 0
            if not np.any(valid_mask):
                continue

            valid_kpts = person_kpts[valid_mask]
            x_min, y_min = np.min(valid_kpts[:, :2], axis=0)
            x_max, y_max = np.max(valid_kpts[:, :2], axis=0)
            bbox = [x_min, y_min, x_max, y_max]

            frame_entry["keypoints"].append(
                {
                    "keypoints": person_kpts[:, :2].tolist(),
                    "scores": person_kpts[:, 2].tolist(),
                    "bboxes": bbox,
                }
            )

        std_formatted.append(frame_entry)

    return std_formatted


def run_detection_pipeline(
    pipeline_config: PipelineConfig,
    global_config: GlobalConfig,
    input_dir: str,
    output_dir: str,
):
    logger.info("Initializing DWpose detection pipeline...")

    device = pipeline_config.processing.device
    max_detections = 30
    detection_thresh = pipeline_config.processing.detection_threshold
    kpt_thresh = pipeline_config.processing.kpt_threshold

    # Initialize models from YAML config paths
    detector = init_detector(
        pipeline_config.models.det_config,
        pipeline_config.models.det_checkpoint,
        device=device,
    )
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)

    pose_estimator = init_pose_model(
        pipeline_config.models.pose_config,
        pipeline_config.models.pose_checkpoint,
        device=device,
    )

    video_files = get_video_files(input_dir, global_config.video.extensions)
    if not video_files:
        logger.warning(f"No video files found in {input_dir}")
        return

    for video_path in video_files:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        rel_path = os.path.relpath(video_path, input_dir)
        current_save_dir = os.path.join(
            output_dir, os.path.dirname(rel_path), video_name
        )
        os.makedirs(current_save_dir, exist_ok=True)

        overlay_video_file = os.path.join(current_save_dir, f"{video_name}_overlay.mp4")

        logger.info(f"Processing: {video_path}")

        # Process video frame by frame
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        predictions = []

        for _ in tqdm(range(frame_count), desc="Processing video"):
            ret, frame = cap.read()
            if not ret:
                break

            # Run detection
            det_results = inference_detector(detector, frame)

            # Filter person detections (class 0 for YOLOX)
            pred_instances = det_results.pred_instances[
                (det_results.pred_instances.labels == 0)
                & (det_results.pred_instances.scores > detection_thresh)
            ]
            bboxes = pred_instances.bboxes.cpu().numpy()[:max_detections]

            if len(bboxes) == 0:
                predictions.append([])
                continue

            # Convert to xywh format
            bboxes_xywh = np.zeros_like(bboxes)
            bboxes_xywh[:, 0] = bboxes[:, 0]  # x
            bboxes_xywh[:, 1] = bboxes[:, 1]  # y
            bboxes_xywh[:, 2] = bboxes[:, 2] - bboxes[:, 0]  # w
            bboxes_xywh[:, 3] = bboxes[:, 3] - bboxes[:, 1]  # h

            # Run pose estimation
            pose_results = inference_topdown(
                pose_estimator, frame, bboxes_xywh, bbox_format="xywh"
            )

            # Collect keypoints for all detected people
            frame_keypoints = []
            for result in pose_results:
                kpts = result.pred_instances.keypoints[0]
                frame_keypoints.append(kpts)

            predictions.append(frame_keypoints)

        cap.release()

        # Convert to standard format
        keypoints_std = convert_to_standard_format(predictions)

        # Convert to VideoData structure
        video_data = VideoData(video_name=video_name)

        # Convert legacy keypoints to VideoData structure
        for frame_idx, frame_data in enumerate(keypoints_std):
            frame_keypoints = frame_data.get("keypoints", [])

            for person_idx, person_data in enumerate(frame_keypoints):
                person = video_data.get_or_create_person(person_idx)

                keypoints = person_data.get("keypoints", [])
                scores = person_data.get("scores", [])
                bbox = person_data.get("bboxes", [])

                # Add detection info
                if bbox:
                    person.add_detection(frame_idx, bbox, 1.0, label=0)

                # Add pose info
                person.add_pose(
                    frame_idx=frame_idx,
                    keypoints=keypoints,
                    keypoints_visible=scores,
                    bbox=bbox,
                    bbox_scores=[1.0] if bbox else [],
                )

        # Set detection config
        detector_config_dict = asdict(pipeline_config.processing)
        video_data.detection_config = detector_config_dict

        # Create SavedData with detection configuration
        saved_data = SavedData.from_video_data(
            video_data,
            detection_config=detector_config_dict,
            processing_metadata={
                "pipeline": "dwpose",
                "input_video": video_path,
                "output_dir": current_save_dir,
            },
        )

        # Save using standard_saver
        save_standard_format(
            data=saved_data,
            output_dir=current_save_dir,
            original_file_path=video_path,
            suffix="",
            save_json=True,
            save_pickle=True,
            save_video_overlay=False,
        )

        logger.info(
            f"Saved keypoints to: {current_save_dir}/{video_name}.json and {current_save_dir}/{video_name}.pkl"
        )

        # Create overlay video (simplified version)
        try:
            cap = cv2.VideoCapture(video_path)
            out = cv2.VideoWriter(
                overlay_video_file,
                cv2.VideoWriter_fourcc(*"mp4v"),
                cap.get(cv2.CAP_PROP_FPS),
                (
                    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                ),
            )

            for frame_idx, frame_keypoints in enumerate(predictions):
                ret, frame = cap.read()
                if not ret:
                    break

                # Draw keypoints and skeleton
                for person_kpts in frame_keypoints:
                    if len(person_kpts) == 0:
                        continue

                    # Check keypoint dimensions before unpacking
                    if person_kpts.shape[1] == 3:  # Keypoints with scores
                        # Draw keypoints
                        for x, y, score in person_kpts:
                            if score > kpt_thresh:
                                cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)

                        # Draw skeleton
                        for i, j in bodyparts2connect:
                            if (
                                i < len(person_kpts)
                                and j < len(person_kpts)
                                and person_kpts[i, 2] > kpt_thresh
                                and person_kpts[j, 2] > kpt_thresh
                            ):
                                pt1 = (int(person_kpts[i, 0]), int(person_kpts[i, 1]))
                                pt2 = (int(person_kpts[j, 0]), int(person_kpts[j, 1]))
                                cv2.line(frame, pt1, pt2, (255, 0, 0), 1)
                    elif person_kpts.shape[1] == 2:  # Keypoints without scores
                        # Draw keypoints (assuming all are valid)
                        for x, y in person_kpts:
                            cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)

                        # Draw skeleton
                        for i, j in bodyparts2connect:
                            if i < len(person_kpts) and j < len(person_kpts):
                                pt1 = (int(person_kpts[i, 0]), int(person_kpts[i, 1]))
                                pt2 = (int(person_kpts[j, 0]), int(person_kpts[j, 1]))
                                cv2.line(frame, pt1, pt2, (255, 0, 0), 1)
                    else:
                        logger.warning(
                            "Unsupported keypoint format, skipping drawing for this person."
                        )

                out.write(frame)

            cap.release()
            out.release()
            logger.info(f"Overlay video saved to: {overlay_video_file}")

        except Exception as e:
            logger.exception(f"Failed to create overlay video: {e}")

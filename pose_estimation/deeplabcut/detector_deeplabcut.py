from pathlib import Path
import torch
import torchvision.models.detection as detection
import numpy as np
from PIL import Image
from deeplabcut.pose_estimation_pytorch import (
    config as dlc_config,
    get_pose_inference_runner
)
from huggingface_hub import hf_hub_download


class DeepLabCutDetector:
    def __init__(self, pipeline_config):
        self.device = pipeline_config.processing.device
        self.max_detections = 10

        # Load DLC detection model
        weights = detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
        self.detector = detection.fasterrcnn_mobilenet_v3_large_fpn(
            weights=weights, box_score_thresh=pipeline_config.processing.detection_threshold
        ).to(self.device).eval()
        self.preprocess = weights.transforms()

        # Load DLC RTMPose
        self.model_dir = Path("hf_files")
        self.model_dir.mkdir(exist_ok=True)
        self.cfg_path = hf_hub_download(
            "DeepLabCut/HumanBody", "rtmpose-x_simcc-body7_pytorch_config.yaml", local_dir=self.model_dir)
        self.pt_path = hf_hub_download(
            "DeepLabCut/HumanBody", "rtmpose-x_simcc-body7.pt", local_dir=self.model_dir)
        self.pose_cfg = dlc_config.read_config_as_dict(self.cfg_path)

        self.runner = get_pose_inference_runner(
            self.pose_cfg,
            snapshot_path=self.pt_path,
            batch_size=16,
            max_individuals=self.max_detections,
        )

    def detect_and_estimate(self, frame):
        # frame: np.ndarray (H x W x 3)
        image_pil = Image.fromarray(frame).convert("RGB")  # only for preprocessing
        image_tensor = self.preprocess(image_pil).to(self.device)

        # Run human detection
        with torch.no_grad():
            preds = self.detector([image_tensor])[0]

        # Get human bounding boxes
        bboxes = [
            box for box, label in zip(preds["boxes"].cpu().numpy(), preds["labels"].cpu().numpy()) if label == 1
        ]
        if not bboxes:
            return []

        bboxes_np = np.stack(bboxes)
        bboxes_xywh = bboxes_np.copy()
        bboxes_xywh[:, 2] -= bboxes_xywh[:, 0]
        bboxes_xywh[:, 3] -= bboxes_xywh[:, 1]

        # Prepare context for RTMPose
        ctx = [{"bboxes": bboxes_xywh[:self.max_detections]}]

        # ✅ Use original NumPy frame, not PIL image
        results = self.runner.inference([(frame, ctx[0])])[0]  # (N, J, 3)
        return results


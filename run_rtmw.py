import mmcv
from mmcv import imread
import mmengine
from mmengine.registry import init_default_scope
import numpy as np
import cv2
import os
import json

from mmpose.apis import inference_topdown
from mmpose.apis import init_model
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

local_runtime = False


def save_keypoints_to_json(pose_results, frame_idx, output_dir):
    keypoint_data = []

    for person in pose_results:
        keypoints = person.pred_instances.keypoints.tolist()
        scores = person.pred_instances.keypoint_scores.tolist()
        keypoint_data.append({
            "keypoints": keypoints,
            "scores": scores
        })

    output_file = os.path.join(output_dir, f"frame_{frame_idx:04d}.json")
    with open(output_file, "w") as f:
        json.dump(keypoint_data, f, indent=4)
    print(f"Saved keypoints of frame {frame_idx} to {output_file}")

# Frame generator for reading video


def frame_generator(video_path):
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        print(f"Error: Couldn't open video {video_path}")
        return

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        yield frame

    video_capture.release()


# Initialize models and visualizer only once before the loop
video_path = "/storage/Projects/Gaitly/bsehgal/HumanEva_walking/Combo_2.avi"

# Detector config and checkpoint
det_config = r"/storage/Projects/Gaitly/bsehgal/RTMW/mmpose/demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py"
det_checkpoint = "https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"
pose_config = r"/storage/Projects/Gaitly/bsehgal/RTMW/mmpose/mmpose/configs/wholebody_2d_keypoint/rtmpose/cocktail13/rtmw-x_8xb704-270e_cocktail14-256x192.py"
pose_checkpoint = r"/storage/Projects/Gaitly/bsehgal/RTMW/rtmw-x_simcc-cocktail14_pt-ucoco_270e-256x192-13a2546d_20231208.pth"
device = "cuda:0"
cfg_options = dict(model=dict(test_cfg=dict(output_heatmaps=True)))

# Initialize detector and pose estimator once
detector = init_detector(det_config, det_checkpoint, device=device)
pose_estimator = init_model(
    pose_config, pose_checkpoint, device=device, cfg_options=cfg_options)

pose_estimator.cfg.visualizer.radius = 3
pose_estimator.cfg.visualizer.line_width = 1
visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)
visualizer.set_dataset_meta(pose_estimator.dataset_meta)

output_dir = r"/storage/Projects/Gaitly/bsehgal/RTMW/output_results"

# Get the scope for detector
scope = detector.cfg.get("default_scope", "mmdet")
if scope is not None:
    init_default_scope(scope)

# Iterate through frames from the video generator
for frame_idx, frame in enumerate(frame_generator(video_path=video_path)):
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detect_result = inference_detector(detector, frame_bgr)
    pred_instance = detect_result.pred_instances.cpu().numpy()
    bboxes = np.concatenate(
        (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1
    )
    bboxes = bboxes[
        np.logical_and(pred_instance.labels == 0, pred_instance.scores > 0.3)
    ]
    bboxes = bboxes[nms(bboxes, 0.3)][:, :4]

    pose_results = inference_topdown(pose_estimator, frame, bboxes)
    data_samples = merge_data_samples(pose_results)

    # Save keypoints to JSON
    save_keypoints_to_json(pose_results, frame_idx, output_dir)

    visualizer.add_datasample(
        "result",
        frame,
        data_sample=data_samples,
        draw_gt=False,
        draw_heatmap=False,
        draw_bbox=False,
        show=False,
        wait_time=0,
        out_file=None,
        kpt_thr=0.3,
    )
    vis_result = visualizer.get_image()
    output_image_file = os.path.join(output_dir, f"frame_{frame_idx:04d}.png")
    mmcv.imwrite(vis_result, output_image_file)
    print(f"Saved visualization of frame {frame_idx} to {output_image_file}")

print("Processing complete. All results are saved in the output directory.")

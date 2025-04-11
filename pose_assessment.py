import os
import pandas as pd
from get_gt_keypoint import extract_ground_truth
from extract_predicted_points import extract_predictions
from compute_pck import compute_pck
from video_info import extract_video_info
import config
from rescale_pred import get_video_resolution, rescale_keypoints

# Define base paths
original_video_base = (
    r"C:\Users\BhavyaSehgal\Downloads\bhavya_1st_sem\humaneva\HumanEva\HumanEva"
)
degraded_video_base = (
    r"C:\Users\BhavyaSehgal\Downloads\bhavya_1st_sem\humaneva\rtmw_x_degraded_40"
)

results = []
lower_body_joints = [
    "LEFT_HIP",
    "RIGHT_HIP",
    "LEFT_KNEE",
    "RIGHT_KNEE",
    "LEFT_ANKLE",
    "RIGHT_ANKLE",
]

for root, _, files in os.walk(degraded_video_base):
    for file in files:
        if file.endswith(".avi"):
            video_info = extract_video_info(file, root)
            if not video_info:
                continue

            subject, action, camera = video_info
            action_group = action.replace(" ", "_")
            cam_name = f"C{camera + 1}"

            print(
                f"Processing: Subject={subject}, Action={action_group}, Camera={cam_name}"
            )

            # Ground truth paths
            original_video_path = os.path.join(
                original_video_base, subject, "Image_Data", file
            )

            # JSON prediction file (same name as .avi but .json)
            json_path = os.path.join(
                root,
                "median",
                os.path.splitext(file)[0] + "_gaussian_filtered_median_filtered.json",
            )
            # json_path = os.path.join(
            #     root,
            #     os.path.splitext(file)[0],
            #     os.path.splitext(file)[0] + ".json",
            # )
            # Extract ground truth keypoints
            gt_keypoints = extract_ground_truth(
                config.CSV_FILE, subject, action, camera
            )
            sync_frame = config.SYNC_DATA.get(subject, {}).get(action, None)[camera]
            frame_range = (sync_frame, sync_frame + len(gt_keypoints))

            # Extract predicted keypoints
            pred_keypoints_org = extract_predictions(json_path, frame_range)

            # Rescale predicted keypoints to match original resolution
            try:
                orig_w, orig_h = get_video_resolution(original_video_path)
                degraded_w, degraded_h = get_video_resolution(os.path.join(root, file))

                scale_x = orig_w / degraded_w
                scale_y = orig_h / degraded_h

                pred_keypoints = rescale_keypoints(pred_keypoints_org, scale_x, scale_y)
            except Exception as e:
                print(f"⚠️ Skipping rescaling due to error: {e}")
                continue

            # Align keypoints
            min_len = min(len(gt_keypoints), len(pred_keypoints))
            gt_keypoints = gt_keypoints[:min_len]
            pred_keypoints = pred_keypoints[:min_len]

            # Compute PCK metrics
            pck2 = compute_pck(
                gt_keypoints,
                pred_keypoints,
                threshold=0.02,
                joints_to_evaluate=lower_body_joints,
            )
            pck1 = compute_pck(
                gt_keypoints,
                pred_keypoints,
                threshold=0.01,
                joints_to_evaluate=lower_body_joints,
            )
            pck5 = compute_pck(
                gt_keypoints,
                pred_keypoints,
                threshold=0.05,
                joints_to_evaluate=lower_body_joints,
            )
            pck05 = compute_pck(
                gt_keypoints,
                pred_keypoints,
                threshold=0.005,
                joints_to_evaluate=lower_body_joints,
            )
            print("--- Results ---")
            print(f"PCK@0.2: {pck2:.2f}%")
            print(f"PCK@0.1: {pck1:.2f}%")
            print(f"PCK@0.05: {pck5:.2f}%")

            results.append([subject, action_group, camera + 1, pck1, pck2, pck5, pck05])

# Save all results
df = pd.DataFrame(
    results,
    columns=[
        "Subject",
        "Action",
        "Camera",
        "PCK@0.01",
        "PCK@0.02",
        "PCK@0.05",
        "PCK@0.005",
    ],
)
excel_path = r"C:\Users\BhavyaSehgal\Downloads\bhavya_1st_sem\humaneva\rtmw_results\comparsion_excels\rtmw_x_40_median_gaussian.xlsx"
df.to_excel(excel_path, index=False)
print(f"Metrics saved to {excel_path}")

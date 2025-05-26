import numpy as np
import pandas as pd
from utils.joint_enum import GTJoints, PredJoints


def compute_jointwise_pck(gt, pred, threshold=0.05, joints_to_evaluate=None):
    """
    Computes PCK per joint for all frames without averaging.
    Returns:
    - joint_names: List of joint names
    - jointwise_pck: (N, J) matrix, where N is frames and J is joints
    """

    # Convert to numpy and check shape
    gt, pred = np.array(gt, dtype=np.float64), np.array(pred, dtype=np.float64)
    if gt.shape[0] != pred.shape[0] or gt.ndim != 3:
        raise ValueError(
            "Shapes of gt and pred must be (N, J, 2) or (N, J, 3)")

    if joints_to_evaluate is None:
        # Full-body evaluation
        joints_to_evaluate = [joint.name for joint in GTJoints]
        right_hip = gt[:, GTJoints.RIGHT_HIP.value]
        left_shoulder = gt[:, GTJoints.LEFT_SHOULDER.value]
        norm_length = np.linalg.norm(left_shoulder - right_hip, axis=-1)
    else:
        left_hip = gt[:, GTJoints.LEFT_HIP.value]
        right_hip = gt[:, GTJoints.RIGHT_HIP.value]
        norm_length = np.linalg.norm(
            right_hip - left_hip, axis=-1)  # Pelvis width

    pred_indices = []
    gt_indices = []
    joint_names = []

    for joint in joints_to_evaluate:
        if joint in PredJoints.__members__:
            gt_joint = GTJoints[joint].value
            pred_joint = PredJoints[joint].value

            if isinstance(gt_joint, tuple):
                gt_indices.append(
                    (gt[:, gt_joint[0]] + gt[:, gt_joint[1]]) / 2)
            else:
                gt_indices.append(gt[:, gt_joint])

            if isinstance(pred_joint, tuple):
                pred_indices.append(
                    (pred[:, pred_joint[0]] + pred[:, pred_joint[1]]) / 2
                )
            else:
                pred_indices.append(pred[:, pred_joint])

            joint_names.append(joint)  # Store joint name

    if not gt_indices or not pred_indices:
        raise ValueError("No valid joints found for evaluation.")

    gt_points = np.stack(gt_indices, axis=1)  # Shape (N, J, 2)
    pred_points = np.stack(pred_indices, axis=1)  # Shape (N, J, 2)

    # Compute distances and correctness per joint, per frame
    distances = (
        np.linalg.norm(gt_points - pred_points, axis=-1) /
        norm_length[:, np.newaxis]
    )
    jointwise_pck = (distances < threshold).astype(int) * 100  # Shape: (N, J)

    return joint_names, jointwise_pck  # No averaging!


# usage_________________________
# # Compute PCK at different thresholds
#             joint_names, jointwise_pck_1 = compute_jointwise_pck(
#                 gt_keypoints,
#                 pred_keypoints,
#                 threshold=0.1,
#                 joints_to_evaluate=lower_body_joints,
#             )
#             _, jointwise_pck_2 = compute_jointwise_pck(
#                 gt_keypoints, pred_keypoints, threshold=0.2
#             )
#             _, jointwise_pck_5 = compute_jointwise_pck(
#                 gt_keypoints, pred_keypoints, threshold=0.5
#             )

#             # Store results per joint
#             for j, joint in enumerate(joint_names):
#                 pck1 = jointwise_pck_1[:, j].mean()  # Average across frames
#                 # pck2 = jointwise_pck_2[:, j].mean()
#                 # pck5 = jointwise_pck_5[:, j].mean()

#                 # Store results as a dictionary where joint names are keys
#                 pck_dict_1 = {
#                     joint: jointwise_pck_1[:, j].mean()
#                     for j, joint in enumerate(joint_names)
#                 }
#                 # pck_dict_2 = {
#                 #     joint: jointwise_pck_2[:, j].mean()
#                 #     for j, joint in enumerate(joint_names)
#                 # }
#                 # pck_dict_5 = {
#                 #     joint: jointwise_pck_5[:, j].mean()
#                 #     for j, joint in enumerate(joint_names)
#                 # }

#                 # Combine results for all joints at different PCK thresholds
#                 results.append(
#                     {
#                         "Subject": subject,
#                         "Action": action_group,
#                         "Camera": camera + 1,
#                         **{
#                             f"{joint}_PCK@0.1": pck_dict_1[joint]
#                             for joint in joint_names
#                         },
#                         # **{f"{joint}_PCK@0.2": pck_dict_2[joint] for joint in joint_names},
#                         # **{f"{joint}_PCK@0.5": pck_dict_5[joint] for joint in joint_names}
#                     }
#                 )

#     # Convert results to DataFrame
#     df_new = pd.DataFrame(results)

#     # Define Excel file path for persistent updates
#     excel_path = r"C:\Users\BhavyaSehgal\Downloads\bhavya_1st_sem\humaneva\rtmw_results\comparsion_excels/rtw_l_lower_aggregated.xlsx"

#     # Check if the Excel file already exists
#     if os.path.exists(excel_path):
#         df_existing = pd.read_excel(excel_path)

#         # Merge the new data with the existing data
#         df_combined = pd.concat([df_existing, df_new], ignore_index=True)

#         # Save updated data back to Excel
#         df_combined.to_excel(excel_path, index=False)
#     else:
#         # If file doesn't exist, create a new one
#         df_new.to_excel(excel_path, index=False)

#     print(f"Updated PCK scores saved to {excel_path}")

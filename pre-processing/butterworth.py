import json
import os
import scipy.signal
from video_info import extract_video_info
import config
from joint_enum import PredJoints


# Function to save filtered keypoints to a new JSON file
def save_filtered_keypoints(output_folder, original_json_path, filtered_keypoints):
    os.makedirs(output_folder, exist_ok=True)  # Ensure the output folder exists
    filtered_json_path = os.path.join(
        output_folder,
        os.path.basename(original_json_path).replace(
            ".json", "_butterworth_filtered.json"
        ),
    )
    with open(filtered_json_path, "w") as f:
        json.dump(filtered_keypoints, f, indent=4)
    print(f"Filtered keypoints saved to: {filtered_json_path}")


base_path = config.VIDEO_FOLDER
lower_body_joints = [
    PredJoints.LEFT_ANKLE.value,
    PredJoints.RIGHT_ANKLE.value,
    PredJoints.LEFT_HIP.value,
    PredJoints.RIGHT_HIP.value,
    PredJoints.LEFT_KNEE.value,
    PredJoints.RIGHT_KNEE.value,
]

# Define the Butterworth filter parameters
order = 4  # Filter order
cutoff = 0.1  # Cutoff frequency, this can be adjusted depending on the frequency of noise in the data

output_base = (
    r"C:\Users\BhavyaSehgal\Downloads\bhavya_1st_sem\humaneva\HumanEva\butterworth"
)

# Walk through all video files in the base directory
for root, dirs, files in os.walk(base_path):
    for file in files:
        video_info = extract_video_info(file, root)
        if video_info:
            subject, action, camera = video_info
            action_group = action.replace(" ", "_")  # Replaces space with underscore
            json_path = os.path.join(
                r"C:\Users\BhavyaSehgal\Downloads\bhavya_1st_sem\humaneva\HumanEva\rtmw_x_degraded",
                subject,
                f"{action_group}_({'C' + str(camera + 1)})",
                f"{action_group}_({'C' + str(camera + 1)})/{action_group}_({'C' + str(camera + 1)})".replace(
                    " ", ""
                )
                + ".json",
            )

            if not os.path.exists(json_path):
                print(f"File not found: {json_path}")
                continue

            # Read JSON directly
            with open(json_path, "r") as f:
                pred_keypoints = json.load(f)

            # Create the Butterworth filter (low-pass filter)
            b, a = scipy.signal.butter(
                order, cutoff, btype="low", fs=30
            )  # fs=30 assumes 30 fps

            num_frames = len(pred_keypoints)
            smoothed_keypoints = [frame.copy() for frame in pred_keypoints]

            for joint_idx in lower_body_joints:
                joint_x_series = [
                    frame["keypoints"][0]["keypoints"][0][joint_idx][0]
                    for frame in pred_keypoints
                ]
                joint_y_series = [
                    frame["keypoints"][0]["keypoints"][0][joint_idx][1]
                    for frame in pred_keypoints
                ]

                # Apply Butterworth filter
                smoothed_x = scipy.signal.filtfilt(b, a, joint_x_series, padlen=4)
                smoothed_y = scipy.signal.filtfilt(b, a, joint_y_series, padlen=4)

                # Update keypoints with filtered values
                for i in range(num_frames):
                    smoothed_keypoints[i]["keypoints"][0]["keypoints"][0][joint_idx] = [
                        float(smoothed_x[i]),
                        float(smoothed_y[i]),
                    ]

            # Define output folder structure
            output_folder = os.path.join(output_base, subject)
            save_filtered_keypoints(output_folder, json_path, smoothed_keypoints)

print("Processing complete.")

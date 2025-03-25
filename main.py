import os
from visualize_pred_points import (
    visualize_predictions,
)  # Import the function from your visualization script
from joint_enum import PredJoints


def main():
    # Set paths for the video and JSON files
    video_path = r"C:\Users\BhavyaSehgal\Downloads\bhavya_1st_sem\humaneva\HumanEva\degraded_videos\S3\Image_Data\Walking_1_(C2).avi"
    json_file = r"C:\Users\BhavyaSehgal\Downloads\bhavya_1st_sem\humaneva\HumanEva\rtmw_x_degraded\S3\Walking_1_(C2)\Walking_1_(C2)\Walking_1_(C2)_gaussian_filtered.json"
    output_video = r"C:\Users\BhavyaSehgal\Downloads\bhavya_1st_sem\humaneva\HumanEva\rtmw_x_degraded\S3\filtered\Walking_1_(C2)\Walking_1_(C2)_gaussian_3.avi"
    # Set frame range (if you want to process specific frames)
    frame_range = None  # None will process all frames

    # Set joint indices to display (None will display all joints)
    joint_indices = [
        PredJoints.LEFT_ANKLE.value,
        PredJoints.RIGHT_ANKLE.value,
        PredJoints.LEFT_HIP.value,
        PredJoints.RIGHT_HIP.value,
        PredJoints.LEFT_KNEE.value,
        PredJoints.RIGHT_KNEE.value,
    ]  # Example: Display only joints with indices 0, 1, 2, 3
    # Or you can set joint_indices = None to display all joints

    # Check if the video and JSON file exist
    if not os.path.exists(video_path):
        print(f"Error: The video file {video_path} was not found.")
        return

    if not os.path.exists(json_file):
        print(f"Error: The JSON file {json_file} was not found.")
        return

    # Call the visualize_predictions function with the provided arguments
    visualize_predictions(
        video_path, json_file, output_video, frame_range, joint_indices
    )


if __name__ == "__main__":
    main()

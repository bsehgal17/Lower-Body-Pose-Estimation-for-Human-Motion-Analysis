import os
import pandas as pd
import matplotlib.pyplot as plt
import cv2


def plot_gt_pred(
    csv_path,
    subject,
    action,
    camera,
    root,
    video_name,
    frame_ranges,
    keypoints_to_plot=None,
):
    """
    Plots ground truth (GT) and predicted keypoints from a CSV file on video frames with point names.

    Parameters:
    - csv_path: Path to the CSV file containing keypoints.
    - subject: Subject ID to filter.
    - action: Action type to filter.
    - camera: Camera ID to filter.
    - root: Path to the video directory.
    - video_name: Name of the video file.
    - frame_ranges: Tuple (start_frame, end_frame) specifying the range of frames to process.
    - keypoints_to_plot: List of keypoint labels to extract (e.g., ['x12', 'y12', 'x13', 'y13']).
                         If None, all keypoints from x1, y1 to x20, y20 are plotted.
    """
    # Load CSV
    df = pd.read_csv(csv_path)

    df["action_group"] = df["Action"].str.extract(r"([a-zA-Z]+\s\d+)")

    # Filter data based on subject, action, and camera
    filtered_df = df[
        (df["Subject"] == subject)
        & (df["action_group"] == action)
        & (df["Camera"] == camera)
        & (df["Action"].str.contains("chunk0"))
    ]

    if filtered_df.empty:
        print("No data found for the given filters.")
        return

    # Define all possible keypoints
    all_keypoints = [f"x{i}" for i in range(1, 21)]
    all_y_points = [f"y{i}" for i in range(1, 21)]

    if keypoints_to_plot is None:
        keypoints_to_plot = all_keypoints + all_y_points  # Plot all if none specified

    # Ensure we get paired keypoints (x, y)
    selected_x_points = [kp for kp in keypoints_to_plot if kp.startswith("x")]
    selected_y_points = [kp.replace("x", "y") for kp in selected_x_points]
    selected_labels = [kp[1:] for kp in selected_x_points]  # Extract numerical label

    # Form the video path
    video_path = os.path.join(root, video_name)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Couldn't open the video.")
        return

    # Loop through frames within the specified range
    for frame_idx, frame_data in filtered_df.iterrows():
        frame_num = frame_ranges[0] + frame_idx
        if frame_num >= frame_ranges[1]:
            break

        # Extract x and y coordinates for the current frame
        x_values = [frame_data[kp] for kp in selected_x_points]
        y_values = [frame_data[kp] for kp in selected_y_points]

        # Set the video to the correct frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

        # Read the frame
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Couldn't read frame {frame_num}.")
            continue

        # Convert the frame from BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Plot the frame
        plt.figure(figsize=(8, 8))
        plt.imshow(frame_rgb)

        # Plot keypoints with labels
        for j, (x, y, label) in enumerate(zip(x_values, y_values, selected_labels)):
            plt.scatter(x, y, c="g", marker="o", s=50, label="GT" if j == 0 else "")
            plt.text(
                x + 3, y, label, fontsize=6, color="g"
            )  # Use numerical point labels

        plt.title(f"Frame {frame_num}")
        plt.legend()
        plt.xlabel("X")
        plt.ylabel("Y")

        # Show the frame
        plt.show()

    # Release the video capture object
    cap.release()

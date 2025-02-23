import cv2
import os
import matplotlib.pyplot as plt


def plot_keypoints_on_frame(frame, gt_keypoints, pred_keypoints):
    """Plot ground truth and predicted keypoints on a video frame."""
    for gt, pred in zip(gt_keypoints, pred_keypoints):
        # Draw ground truth points in red
        cv2.circle(frame, (int(gt[0]), int(gt[1])), 5, (0, 0, 255), -1)
        # Draw predicted points in blue
        cv2.circle(frame, (int(pred[0]), int(pred[1])), 5, (255, 0, 0), -1)
    return frame


def process_video(root_path, video_filename, gt_keypoints, pred_keypoints, start_frame=0, end_frame=None):
    """Check if lengths match and plot keypoints on video frames."""

    # Join the root path and video filename to get the full video path
    video_path = os.path.join(root_path, video_filename)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video at {video_path}")
        return

    # Check if the lengths of ground truth and predicted keypoints are the same
    if len(gt_keypoints) != len(pred_keypoints):
        print("Error: Ground truth and predicted keypoints lengths do not match!")
        return

    # Set the frame range to start from 'start_frame' and end at 'end_frame' (if provided)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_idx = start_frame

    while True:
        ret, frame = cap.read()

        if not ret:
            break  # End of video

        # Stop the loop once we've reached the end_frame (if it's provided)
        if end_frame is not None and frame_idx >= end_frame:
            break

        # If the current frame index is less than the length of keypoints
        if frame_idx < len(gt_keypoints):
            gt = gt_keypoints[frame_idx]
            pred = pred_keypoints[frame_idx]

            # Plot the keypoints on the current frame
            frame_with_keypoints = plot_keypoints_on_frame(frame, gt, pred)

            # Convert the frame to RGB (matplotlib uses RGB instead of BGR)
            frame_with_keypoints_rgb = cv2.cvtColor(
                frame_with_keypoints, cv2.COLOR_BGR2RGB)

            # Display the frame with keypoints using matplotlib
            plt.imshow(frame_with_keypoints_rgb)
            plt.axis('off')  # Hide axes
            plt.show(block=False)  # Show without blocking the code
            plt.pause(0.1)  # Pause to allow frame to stay for a moment

            # Wait for the window to close before continuing to the next frame
            plt.waitforbuttonpress()  # Wait for any button press to move to the next frame

        frame_idx += 1

    # Release video capture object
    cap.release()

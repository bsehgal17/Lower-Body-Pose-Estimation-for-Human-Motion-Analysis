import os
import matplotlib.pyplot as plt
import cv2



def plot_gt_pred(gt, pred, root, video_name, frame_ranges):
    # Form the video path
    video_path = os.path.join(root, video_name)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Couldn't open the video.")
        return

    # Loop through the frame range
    for i, frame_num in enumerate(frame_ranges):  # Loop through frame_ranges and get frame number
        while frame_num<frame_ranges[1]:
            # Set the video to the correct frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

            # Read the frame
            ret, frame = cap.read()
            if not ret:
                print(f"Error: Couldn't read frame {frame_num}.")
                continue
            
            # Convert the frame from BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Get the GT and Pred points using the index (i) in frame_ranges
            gt_points = gt[i] if i < len(gt) else None
            pred_points = pred[i] if i < len(pred) else None
            
            # Plot the frame
            plt.figure(figsize=(8, 8))
            plt.imshow(frame_rgb)

            # Plot GT points (green)
            if gt_points is not None:
                plt.scatter(gt_points[:, 0], gt_points[:, 1], c='g', label='GT', marker='o', s=50)
            
            # Plot Pred points (red)
            if pred_points is not None:
                plt.scatter(pred_points[:, 0], pred_points[:, 1], c='r', label='Pred', marker='x', s=50)

            plt.title(f"Frame {frame_num}")
            plt.legend()
            plt.xlabel("X")
            plt.ylabel("Y")
            
            # Show the frame and wait until the window is closed
            plt.show()
            frame_num = frame_num+1
            i=i+1

        # Release the video capture object
        cap.release()
        
import os
import cv2
import numpy as np
from Compute_metrics.get_gt_keypoint import extract_ground_truth
from utils.video_info import extract_video_info
import config

sync_data = {
    'S1': {
        'Walking 1': (82, 81, 82),
        'Jog 1': (51, 51, 50),

    },
    'S2': {
        'Walking 1': (115, 115, 114),
        'Jog 1': (100, 100, 99),

    },
    'S3': {
        'Walking 1': (80, 80, 80),
        'Jog 1': (65, 65, 65),

    },

}


def process_all_videos(csv_path, base_path):
    for root, dirs, files in os.walk(base_path):
        for file in files:
            video_info = extract_video_info(file, root)
            if video_info:
                subject, action_group, camera = video_info
                video_path = os.path.join(root, file)

                print(
                    f"Processing video: Subject={subject}, Action={action_group}, Camera={camera + 1}")

                # Get keypoints for the given subject, action_group, and camera
                keypoints = extract_ground_truth(
                    csv_path, subject, action_group, camera)

                if keypoints:
                    # Open the video file and process each frame
                    cap = cv2.VideoCapture(video_path)

                    if not cap.isOpened():
                        print(f"Error: Could not open video {video_path}")
                        continue

                    sync_frame = sync_data[subject][action_group][camera]
                    frame_number = int(sync_frame)

                    # Define output video file
                    output_video_filename = f"{action_group.replace(' ', '_')}_C{camera + 1}_output.avi"
                    output_video_dir = os.path.join(
                        os.path.dirname(video_path), "Output_Videos")
                    os.makedirs(output_video_dir, exist_ok=True)

                    output_video_path = os.path.join(
                        output_video_dir, output_video_filename)

                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    out = cv2.VideoWriter(
                        output_video_path, fourcc, fps, (frame_width, frame_height))

                    for keypoint_row in keypoints:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                        ret, frame = cap.read()
                        if not ret:
                            print(
                                f"Error: Could not read frame {frame_number} for video {video_path}")
                            continue

                        # Overlay keypoints on the frame
                        for (x, y) in keypoint_row:
                            if not np.isnan(x) and not np.isnan(y):
                                cv2.circle(frame, (int(x), int(y)),
                                           5, (0, 0, 255), -1)

                        # Optionally, draw skeleton joints (not included in this example)

                        # Write the frame to the output video
                        out.write(frame)
                        frame_number += 1

                    cap.release()
                    print(
                        f"Processed and saved output video: {output_video_path}")
                else:
                    print(
                        f"No keypoints found for Subject={subject}, Action={action_group}, Camera={camera + 1}")

                print("-" * 50)


process_all_videos(csv_path=config.CSV_FILE, base_path=config.VIDEO_FOLDER)

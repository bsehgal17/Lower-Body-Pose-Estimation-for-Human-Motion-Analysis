import cv2
import os
from config import VIDEO_FOLDER, OUTPUT_DIR
from utils import get_video_files, frame_generator, save_keypoints_to_json
from detector import init_object_detector, detect_humans
from pose_estimator import init_pose_estimator, estimate_pose
from visualization import init_visualizer, visualize_pose

# Initialize models
detector = init_object_detector()
pose_estimator = init_pose_estimator()
visualizer = init_visualizer(pose_estimator)

# Get all videos from the folder
video_files = get_video_files(VIDEO_FOLDER)
if not video_files:
    print("No video files found in the folder!")
    exit()

for video_path in video_files:
    video_name = os.path.basename(video_path)
    print(f"\nProcessing Video: {video_name}")

    for frame_idx, frame in enumerate(frame_generator(video_path)):
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect humans
        bboxes = detect_humans(detector, frame_bgr)

        # Pose estimation
        data_samples, pose_results = estimate_pose(pose_estimator, frame, bboxes)

        # Save results
        save_keypoints_to_json(pose_results, frame_idx, OUTPUT_DIR, video_name)
        visualize_pose(frame, data_samples, visualizer, frame_idx)

print("Processing complete. All results are saved in the output directory.")

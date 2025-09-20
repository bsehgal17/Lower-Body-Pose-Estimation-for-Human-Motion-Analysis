import os
import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime


class MediaPipePoseEstimator:
    def __init__(self, static_image_mode=False, model_complexity=1,
                 smooth_landmarks=True, min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def process_video(self, input_path, output_video_path=None, output_data_path=None):
        """Process a single video and return pose estimation results"""

        # Read video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {input_path}")
            return None

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Initialize video writer if output path is provided
        if output_video_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(
                output_video_path, fourcc, fps, (width, height))
        else:
            out = None

        # Store pose data
        pose_data = {
            'video_path': input_path,
            'fps': fps,
            'resolution': (width, height),
            'total_frames': total_frames,
            'frames': []
        }

        # Process each frame
        frame_count = 0
        pbar = tqdm(total=total_frames,
                    desc=f"Processing {os.path.basename(input_path)}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process frame with MediaPipe
            results = self.pose.process(rgb_frame)

            # Extract pose landmarks
            frame_data = {
                'frame_number': frame_count,
                'timestamp': frame_count / fps,
                'landmarks': None,
                'world_landmarks': None
            }

            if results.pose_landmarks:
                # Convert landmarks to serializable format
                landmarks = []
                for idx, landmark in enumerate(results.pose_landmarks.landmark):
                    landmarks.append({
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z,
                        'visibility': landmark.visibility
                    })
                frame_data['landmarks'] = landmarks

            if results.pose_world_landmarks:
                world_landmarks = []
                for idx, landmark in enumerate(results.pose_world_landmarks.landmark):
                    world_landmarks.append({
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z,
                        'visibility': landmark.visibility
                    })
                frame_data['world_landmarks'] = world_landmarks

            pose_data['frames'].append(frame_data)

            # Draw landmarks on frame if output video is requested
            if out and results.pose_landmarks:
                annotated_frame = frame.copy()
                self.mp_drawing.draw_landmarks(
                    annotated_frame,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                )
                out.write(annotated_frame)

            frame_count += 1
            pbar.update(1)

        pbar.close()
        cap.release()
        if out:
            out.release()

        # Save pose data
        if output_data_path:
            with open(output_data_path, 'w') as f:
                json.dump(pose_data, f, indent=2)

        return pose_data


def find_videos(directory, extensions=['.mp4', '.avi', '.mov', '.mkv']):
    """Recursively find all video files in directory and subdirectories"""
    video_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                video_files.append(os.path.join(root, file))
    return video_files


def process_all_videos(input_directory, output_base_dir):
    """Process all videos in nested directories"""

    # Create output directories
    processed_videos_dir = os.path.join(output_base_dir, 'processed_videos')
    pose_data_dir = os.path.join(output_base_dir, 'pose_data')

    os.makedirs(processed_videos_dir, exist_ok=True)
    os.makedirs(pose_data_dir, exist_ok=True)

    # Find all video files
    video_files = find_videos(input_directory)
    print(f"Found {len(video_files)} video files to process")

    # Initialize pose estimator
    pose_estimator = MediaPipePoseEstimator()

    results_summary = []

    for video_path in video_files:
        try:
            # Create output paths preserving directory structure
            rel_path = os.path.relpath(video_path, input_directory)
            video_name = os.path.splitext(rel_path)[0]

            output_video_path = os.path.join(
                processed_videos_dir, f"{video_name}_processed.mp4")
            output_data_path = os.path.join(
                pose_data_dir, f"{video_name}_pose_data.json")

            # Create necessary subdirectories
            os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
            os.makedirs(os.path.dirname(output_data_path), exist_ok=True)

            print(f"\nProcessing: {rel_path}")

            # Process video
            pose_data = pose_estimator.process_video(
                video_path,
                output_video_path,
                output_data_path
            )

            if pose_data:
                results_summary.append({
                    'input_path': video_path,
                    'output_video': output_video_path,
                    'output_data': output_data_path,
                    'total_frames': pose_data['total_frames'],
                    'success': True
                })

        except Exception as e:
            print(f"Error processing {video_path}: {str(e)}")
            results_summary.append({
                'input_path': video_path,
                'error': str(e),
                'success': False
            })

    # Save processing summary
    summary_path = os.path.join(
        output_base_dir, f"processing_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=2)

    return results_summary


def main():
    # Change this to your input directory
    input_directory = "/storage/Projects/Gaitly/bsehgal/lower_body_pose_est/HumanEva_Walk"
    # Change this to your desired output directory
    output_directory = "/storage/Projects/Gaitly/bsehgal/lower_body_pose_est/pipeline_results/HumanEva/Mediapipe"

    print("Starting MediaPipe Pose Estimation on Videos")
    print("=" * 50)

    results = process_all_videos(input_directory, output_directory)

    # Print summary
    successful = sum(1 for r in results if r['success'])
    print(f"\nProcessing completed!")
    print(f"Successful: {successful}/{len(results)}")
    print(f"Failed: {len(results) - successful}")

    if successful > 0:
        print(f"\nResults saved in: {output_directory}")


if __name__ == "__main__":
    main()

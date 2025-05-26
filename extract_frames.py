import os
import cv2


def extract_frames_from_video(video_path, output_folder):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_name = f"frame_{frame_count:05d}.jpg"
        frame_path = os.path.join(output_folder, frame_name)
        cv2.imwrite(frame_path, frame)
        frame_count += 1

    cap.release()
    print(f"Extracted {frame_count} frames from {video_path} to {output_folder}")


def process_all_videos(input_root, output_root):
    supported_exts = (".mp4", ".avi", ".mov", ".mkv")
    for dirpath, _, filenames in os.walk(input_root):
        for filename in filenames:
            if filename.lower().endswith(supported_exts):
                video_path = os.path.join(dirpath, filename)

                # Create mirrored folder structure
                relative_path = os.path.relpath(dirpath, input_root)
                output_folder = os.path.join(
                    output_root, relative_path, os.path.splitext(filename)[0]
                )
                os.makedirs(output_folder, exist_ok=True)

                extract_frames_from_video(video_path, output_folder)


if __name__ == "__main__":
    input_folder = r"C:\Users\BhavyaSehgal\Downloads\bhavya_phd\test_dataset_results\degraded_videos"  # Replace this with your input folder
    output_folder = r"C:\Users\BhavyaSehgal\Downloads\bhavya_phd\video frames\degraded"  # Replace this with where you want frames saved

    process_all_videos(input_folder, output_folder)

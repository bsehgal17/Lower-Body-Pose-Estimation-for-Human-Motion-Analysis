import cv2
import os
import pandas as pd
import matplotlib.pyplot as plt


def overlay_gt_on_frames(video_root, csv_file, output_root):
    os.makedirs(output_root, exist_ok=True)

    # Load the single GT CSV file
    df = pd.read_csv(csv_file)

    for root, _, files in os.walk(video_root):
        for video_file in files:
            if not video_file.endswith(".avi"):
                continue

            video_path = os.path.join(root, video_file)

            parts = video_path.split(os.sep)
            subject = parts[-3]  # e.g. "S2"
            action = video_file.split("_")[0]  # e.g. "Walking"
            camera_str = video_file.split("(C")[-1].split(")")[0]  # e.g. "3"
            camera = int(camera_str)

            print(
                f"Processing {video_file} | Subject={subject}, Action={action}, Camera={camera}"
            )

            # Find first GT row matching this video
            subset = df[
                (df["Subject"] == subject)
                & (df["Action"].str.contains(action))
                & (df["Camera"] == camera)
            ]
            if subset.empty:
                print(f"⚠️ No GT found for {video_file}")
                continue

            first_row = subset.iloc[0]

            # Extract keypoints using columns named x1, y1, x2, y2, ...
            keypoints = []
            idx = 1
            while True:
                x_col = f"x{idx}"
                y_col = f"y{idx}"
                if x_col in first_row and y_col in first_row:
                    try:
                        x = int(float(first_row[x_col]))
                        y = int(float(first_row[y_col]))
                        keypoints.append((x, y))
                    except (ValueError, TypeError):
                        pass
                    idx += 1
                else:
                    break

            # Prepare output folder for frames
            rel_path = os.path.relpath(root, video_root)
            out_folder = os.path.join(
                output_root, rel_path, os.path.splitext(video_file)[0]
            )
            os.makedirs(out_folder, exist_ok=True)

            # Show first frame of video with keypoints using matplotlib
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                plt.figure(figsize=(8, 6))
                plt.imshow(frame_rgb)
                if keypoints:
                    xs, ys = zip(*keypoints)
                    plt.scatter(xs, ys, c="lime", s=40, edgecolors="black")
                plt.title(f"First frame: {video_file}")
                plt.axis("off")
                plt.show()
            cap.release()

            cap = cv2.VideoCapture(video_path)
            frame_idx = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Overlay same GT keypoints on all frames
                for x, y in keypoints:
                    cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

                # Save frame as image
                frame_path = os.path.join(out_folder, f"frame_{frame_idx:04d}.png")
                cv2.imwrite(frame_path, frame)
                frame_idx += 1

            cap.release()
            print(f"✅ Saved frames to: {out_folder}")


# Example usage
overlay_gt_on_frames(
    video_root=r"C:\Users\BhavyaSehgal\Downloads\HumanEva",
    csv_file=r"C:\Users\BhavyaSehgal\Downloads\HumanEva\validate_combined_chunk0.csv",
    output_root=r"C:\Users\BhavyaSehgal\Downloads\output_frames_new",
)

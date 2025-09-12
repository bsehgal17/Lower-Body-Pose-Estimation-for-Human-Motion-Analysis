import cv2
import os
import pandas as pd


def overlay_gt_on_videos(video_root, csv_file, output_root):
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
            camera = int(camera_str)  # shift (3 -> 2, 2 -> 1, etc.)

            print(
                f"Processing {video_file} | Subject={subject}, Action={action}, Camera={camera}"
            )

            # Find rows in CSV matching this video
            subset = df[
                (df["Subject"] == subject)
                & (df["Action"].str.contains(action))
                & (df["Camera"] == camera)
            ]
            if subset.empty:
                print(f"⚠️ No GT found for {video_file}")
                continue

            # Take first frame's GT data
            first_row = subset.iloc[0]

            # Extract only numeric values (ignore non-numeric like "Box 1")
            numeric_values = []
            for v in first_row.values:
                try:
                    f = float(v)
                    numeric_values.append(f)
                except ValueError:
                    continue  # skip non-numeric

            # Skip the first numeric value (Camera)
            numeric_values = numeric_values[1:]

            # Convert into (x,y) pairs
            keypoints = [
                (int(numeric_values[i]), int(numeric_values[i + 1]))
                for i in range(0, len(numeric_values), 2)
            ]

            # Prepare output folder structure
            rel_path = os.path.relpath(root, video_root)
            out_folder = os.path.join(output_root, rel_path)
            os.makedirs(out_folder, exist_ok=True)
            out_path = os.path.join(
                out_folder, f"{os.path.splitext(video_file)[0]}.avi"
            )

            cap = cv2.VideoCapture(video_path)
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            out = cv2.VideoWriter(
                out_path,
                fourcc,
                cap.get(cv2.CAP_PROP_FPS),
                (int(cap.get(3)), int(cap.get(4))),
            )

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Overlay GT keypoints
                for x, y in keypoints:
                    cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

                out.write(frame)

            cap.release()
            out.release()
            print(f"✅ Saved: {out_path}")


# Example usage
overlay_gt_on_videos(
    video_root=r"C:\Users\BhavyaSehgal\Downloads\HumanEva",
    csv_file=r"C:\Users\BhavyaSehgal\Downloads\HumanEva\validate_combined_chunk0.csv",
    output_root=r"C:\Users\BhavyaSehgal\Downloads\output",
)

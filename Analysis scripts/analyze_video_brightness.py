import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

# --- USER CONFIGURATION ---
FOLDER_PATH = r"C:\Users\BhavyaSehgal\Downloads\evidence for brightness"
# ---------------------------


def analyze_brightness_single(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return None

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30  # fallback to 30 if fps not detected
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    print(f"Video: {video_name}, FPS: {fps}")

    brightness_values = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % fps == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            avg_brightness = np.mean(gray)
            brightness_values.append(avg_brightness)

        frame_idx += 1

    cap.release()

    brightness_values = np.array(brightness_values)
    mean_brightness = np.mean(brightness_values)
    min_brightness = np.min(brightness_values)
    max_brightness = np.max(brightness_values)

    # Plot brightness over time
    plt.figure(figsize=(10, 4))
    plt.plot(brightness_values, label="Brightness")
    plt.xlabel("Sampled Frame Index (1 per sec)")
    plt.ylabel("Avg Brightness")
    plt.title(f"Brightness Over Time - {video_name}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    time_plot_path = os.path.join(output_dir, f"brightness_over_time_{video_name}.png")
    plt.savefig(time_plot_path)
    plt.close()

    # Plot histogram
    plt.figure(figsize=(6, 4))
    plt.hist(brightness_values, bins=30, color="gray", edgecolor="black")
    plt.xlabel("Brightness Value")
    plt.ylabel("Frequency")
    plt.title(f"Brightness Histogram - {video_name}")
    plt.tight_layout()
    hist_plot_path = os.path.join(output_dir, f"brightness_histogram_{video_name}.png")
    plt.savefig(hist_plot_path)
    plt.close()

    return {
        "video": video_name,
        "fps": fps,
        "mean_brightness": round(mean_brightness, 2),
        "min_brightness": round(min_brightness, 2),
        "max_brightness": round(max_brightness, 2),
        "time_plot": time_plot_path,
        "histogram_plot": hist_plot_path,
    }


def analyze_brightness_folder(folder_path):
    output_dir = os.path.join(folder_path, "brightness_outputs")
    os.makedirs(output_dir, exist_ok=True)

    results = []
    video_files = [
        f
        for f in os.listdir(folder_path)
        if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
    ]

    if not video_files:
        print("No video files found in the folder.")
        return

    for video_file in video_files:
        full_path = os.path.join(folder_path, video_file)
        print(f"\nProcessing: {video_file}")
        result = analyze_brightness_single(full_path, output_dir)
        if result:
            results.append(result)

    # Save summary CSV
    summary_df = pd.DataFrame(results)
    summary_csv_path = os.path.join(output_dir, "brightness_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)

    print("Analysis complete.")
    print(f"Summary CSV: {summary_csv_path}")
    print(f"Plots saved in: {output_dir}")


# Run analysis
if __name__ == "__main__":
    analyze_brightness_folder(FOLDER_PATH)

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from tqdm import tqdm

# --- USER CONFIGURATION ---
FOLDER_PATH = r"C:\Users\BhavyaSehgal\Downloads\evidence for brightness"
OUTPUT_DIR_NAME = "brightness_analysis"  # Separate output folder name
REGIONAL_ANALYSIS = True  # Set False to disable regional brightness analysis
# ---------------------------


def calculate_perceptual_luminance(frame):
    """Calculate perceptual luminance using the standard ITU-R BT.709 formula"""
    if frame is None or frame.size == 0:
        return None

    # Convert to float for accurate calculations
    frame = frame.astype("float32") / 255.0

    # Gamma expansion (inverse of sRGB gamma)
    mask = frame <= 0.04045
    frame[mask] = frame[mask] / 12.92
    frame[~mask] = ((frame[~mask] + 0.055) / 1.055) ** 2.4

    # Calculate luminance (ITU-R BT.709 coefficients)
    b, g, r = cv2.split(frame)
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b

    # Convert back to 0-255 range
    return (luminance * 255).astype("uint8")


def analyze_frame(frame, frame_size):
    """Analyze a single frame with multiple brightness metrics"""
    results = {}

    if frame is None:
        return None, None

    # Convert to perceptual luminance
    luminance = calculate_perceptual_luminance(frame)
    if luminance is None:
        return None, None

    # Global brightness metrics
    results["global_mean"] = np.mean(luminance)
    results["global_median"] = np.median(luminance)
    results["global_std"] = np.std(luminance)

    if REGIONAL_ANALYSIS:
        # Divide frame into 3x3 grid
        h, w = luminance.shape
        regions = []
        for i in range(3):
            for j in range(3):
                x1, x2 = w // 3 * j, w // 3 * (j + 1)
                y1, y2 = h // 3 * i, h // 3 * (i + 1)
                region = luminance[y1:y2, x1:x2]
                regions.append(
                    {
                        "position": f"{i + 1}-{j + 1}",
                        "mean": np.mean(region),
                        "median": np.median(region),
                        "std": np.std(region),
                    }
                )
        results["regions"] = regions

    # Histogram analysis (0-255 bins)
    hist = cv2.calcHist([luminance], [0], None, [256], [0, 256])
    results["histogram"] = hist.flatten()

    return results, luminance


def analyze_video(video_path, output_dir):
    """Analyze a single video file"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return None

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    print(f"\nAnalyzing: {video_name}")
    print(f"Duration: {total_frames / fps:.1f}s, FPS: {fps}, Frames: {total_frames}")

    # Initialize data storage
    frame_data = []
    sampled_luminance = []
    sampling_rate = max(1, fps)  # Sample once per second

    # Process frames with progress bar
    for frame_idx in tqdm(range(total_frames), desc="Processing frames"):
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sampling_rate == 0:
            frame_results, luminance = analyze_frame(
                frame, (frame.shape[1], frame.shape[0])
            )
            if frame_results is not None and luminance is not None:
                frame_results["frame_idx"] = frame_idx
                frame_results["timestamp"] = frame_idx / fps
                frame_data.append(frame_results)
                sampled_luminance.append(luminance)

    cap.release()

    if not frame_data:
        return None

    # Aggregate results
    global_means = [f["global_mean"] for f in frame_data]
    timestamps = [f["timestamp"] for f in frame_data]

    # Calculate video-level statistics
    video_stats = {
        "video_name": video_name,
        "duration_sec": total_frames / fps,
        "fps": fps,
        "mean_brightness": np.mean(global_means),
        "median_brightness": np.median(global_means),
        "std_brightness": np.std(global_means),
        "min_brightness": np.min(global_means),
        "max_brightness": np.max(global_means),
        "brightness_range": np.ptp(global_means),
        "frame_count": len(frame_data),
    }

    # Add regional stats if enabled and available
    if REGIONAL_ANALYSIS and "regions" in frame_data[0]:
        regions = ["1-1", "1-2", "1-3", "2-1", "2-2", "2-3", "3-1", "3-2", "3-3"]
        for region_pos in regions:
            region_means = []
            for f in frame_data:
                if "regions" in f:
                    for region in f["regions"]:
                        if region["position"] == region_pos:
                            region_means.append(region["mean"])
                            break

            if region_means:  # Only add if we found data
                video_stats[f"region_{region_pos}_mean"] = np.mean(region_means)
                video_stats[f"region_{region_pos}_std"] = np.std(region_means)

    # Visualization
    os.makedirs(output_dir, exist_ok=True)
    plot_brightness_over_time(timestamps, global_means, video_name, output_dir)
    plot_brightness_histogram(global_means, video_name, output_dir)

    if REGIONAL_ANALYSIS and len(sampled_luminance) > 0:
        plot_regional_brightness(sampled_luminance[-1], video_name, output_dir)

    return video_stats


def plot_brightness_over_time(timestamps, brightness_values, video_name, output_dir):
    plt.figure(figsize=(12, 5))
    plt.plot(timestamps, brightness_values, label="Perceptual Brightness")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Brightness (0-255)")
    plt.title(f"Brightness Over Time - {video_name}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{video_name}_brightness_timeseries.png"))
    plt.close()


def plot_brightness_histogram(brightness_values, video_name, output_dir):
    plt.figure(figsize=(8, 5))
    plt.hist(brightness_values, bins=30, color="skyblue", edgecolor="black")
    plt.xlabel("Brightness Value")
    plt.ylabel("Frequency")
    plt.title(f"Brightness Distribution - {video_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{video_name}_brightness_histogram.png"))
    plt.close()


def plot_regional_brightness(luminance_frame, video_name, output_dir):
    plt.figure(figsize=(10, 8))
    plt.imshow(luminance_frame, cmap="gray")
    plt.colorbar(label="Brightness")

    # Draw grid lines
    h, w = luminance_frame.shape
    for i in range(1, 3):
        plt.axhline(y=i * h / 3, color="red", linestyle="--", linewidth=1)
        plt.axvline(x=i * w / 3, color="red", linestyle="--", linewidth=1)

    plt.title(f"Regional Brightness - {video_name}\n(Sampled Frame)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{video_name}_regional_brightness.png"))
    plt.close()


def analyze_video_folder(folder_path):
    """Analyze all videos in a folder and its subfolders"""
    output_dir = os.path.join(folder_path, OUTPUT_DIR_NAME)
    os.makedirs(output_dir, exist_ok=True)

    # Find all video files
    video_files = []
    for root, _, files in os.walk(folder_path):
        for f in files:
            if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".flv")):
                # Skip files in our output directory
                if not root.startswith(os.path.join(folder_path, OUTPUT_DIR_NAME)):
                    video_files.append(os.path.join(root, f))

    if not video_files:
        print(f"No video files found in {folder_path}")
        return

    print(f"Found {len(video_files)} video files to analyze")

    # Process all videos
    all_results = []
    for video_path in video_files:
        result = analyze_video(video_path, output_dir)
        if result:
            all_results.append(result)

    # Save comprehensive results
    if all_results:
        df = pd.DataFrame(all_results)
        summary_path = os.path.join(output_dir, "brightness_summary.csv")
        df.to_csv(summary_path, index=False)
        print(f"\nAnalysis complete. Results saved to:\n{summary_path}")

        # Generate a quick summary report
        with open(os.path.join(output_dir, "summary_report.txt"), "w") as f:
            f.write("Video Brightness Analysis Summary\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Analyzed {len(all_results)} videos\n\n")

            global_means = df["mean_brightness"]
            f.write(f"Overall brightness statistics:\n")
            f.write(f"- Average brightness: {np.mean(global_means):.1f}\n")
            f.write(
                f"- Brightness range: {np.min(global_means):.1f} to {np.max(global_means):.1f}\n"
            )
            f.write(f"- Standard deviation: {np.std(global_means):.1f}\n\n")

            if REGIONAL_ANALYSIS:
                f.write("Regional brightness variations:\n")
                regions = [
                    "1-1",
                    "1-2",
                    "1-3",
                    "2-1",
                    "2-2",
                    "2-3",
                    "3-1",
                    "3-2",
                    "3-3",
                ]
                for region in regions:
                    col_name = f"region_{region}_mean"
                    if col_name in df.columns:
                        f.write(
                            f"- Region {region}: Avg {df[col_name].mean():.1f} ± {df[col_name].std():.1f}\n"
                        )


if __name__ == "__main__":
    analyze_video_folder(FOLDER_PATH)

#!/usr/bin/env python3
"""
AlphaPose Batch Video Processing Script
Processes all videos in a folder recursively using AlphaPose
"""

import os
import glob
import subprocess
import sys


def run_command(cmd, description=""):
    """Run a shell command with error handling"""
    if description:
        print(f"\n{description}")
        print("-" * 50)

    print(f"Running: {cmd}")
    try:
        result = subprocess.run(
            cmd, shell=True, check=True, capture_output=True, text=True
        )
        print("✓ Command executed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Command failed: {e}")
        print(f"Error output: {e.stderr[:500]}...")
        return False


def main():
    # Set working directory
    os.chdir("/content/AlphaPose")

    # Input and output directories
    input_folder = "/content/drive/MyDrive/HumanEva_walk"
    output_folder = "examples/res"

    # Get all video files recursively
    video_files = []
    for ext in ["*.avi", "*.mp4", "*.mov"]:
        video_files.extend(
            glob.glob(os.path.join(input_folder, "**", ext), recursive=True)
        )

    print(f"Found {len(video_files)} videos to process")

    # Process each video
    for video_path in video_files:
        relative_path = os.path.relpath(video_path, input_folder)
        video_dir = os.path.dirname(relative_path)
        video_output_dir = os.path.join(output_folder, video_dir)
        os.makedirs(video_output_dir, exist_ok=True)

        print(f"\nProcessing: {relative_path}")

        cmd = f"""
        PYTHONPATH=/content/AlphaPose:$PYTHONPATH python scripts/demo_inference.py \
            --cfg configs/halpe_26/resnet/256x192_res50_lr1e-3_1x.yaml \
            --checkpoint pretrained_models/halpe26_fast_res50_256x192.pth \
            --video "{video_path}" \
            --outdir "{video_output_dir}" \
            --save_video \
            --detector yolo \
            --sp \
            --gpus -1
        """

        success = run_command(cmd, f"Processing {os.path.basename(video_path)}")
        if not success:
            print(f"Failed to process {video_path}, continuing with next video...")

    # Show results
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETED")
    print("=" * 60)

    result_files = glob.glob(os.path.join(output_folder, "**", "*"), recursive=True)
    for file in result_files:
        if os.path.isfile(file):
            print(f"📄 {file}")


if __name__ == "__main__":
    main()

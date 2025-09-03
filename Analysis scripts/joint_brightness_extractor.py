"""
Joint Brightness Extractor Script

Extracts brightness values at specific joint coordinates from video frames.
Focus: Brightness extraction at joint coordinates only.
"""

import sys
import os
import cv2
import numpy as np
import pandas as pd
from typing import List, Dict

# Add the Analysis scripts directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gt_data_loader import GroundTruthDataLoader


class JointBrightnessExtractor:
    """Extract brightness values at joint coordinate locations."""

    def __init__(self, dataset_name: str):
        """Initialize with dataset name."""
        self.dataset_name = dataset_name
        self.gt_loader = GroundTruthDataLoader(dataset_name)
        self._load_config()

    def _load_config(self):
        """Load video and dataset configuration."""
        try:
            from config import ConfigManager

            self.config = ConfigManager.load_config(self.dataset_name)

            # Get video folder path
            if hasattr(self.config, "video_folder"):
                self.video_folder = self.config.video_folder
            elif hasattr(self.config.paths, "video_folder"):
                self.video_folder = self.config.paths.video_folder
            else:
                print("⚠️  No video folder configured, will need video path")
                self.video_folder = None

            print(f"✅ Video config loaded for {self.dataset_name}")

        except Exception as e:
            print(f"❌ Failed to load video config: {e}")
            raise

    def get_video_files(self, video_folder: str = None) -> List[str]:
        """Get list of video files in the folder."""
        if video_folder is None:
            video_folder = self.video_folder

        if not video_folder or not os.path.exists(video_folder):
            print(f"❌ Video folder not found: {video_folder}")
            return []

        video_extensions = (".mp4", ".avi", ".mov", ".mkv", ".wmv")
        video_files = []

        for root, dirs, files in os.walk(video_folder):
            for file in files:
                if file.lower().endswith(video_extensions):
                    video_files.append(os.path.join(root, file))

        print(f"✅ Found {len(video_files)} video files")
        return video_files

    def extract_brightness_at_coordinates(
        self,
        video_path: str,
        coordinates: Dict[str, np.ndarray],
        sampling_radius: int = 3,
        frame_limit: int = None,
    ) -> Dict[str, List[float]]:
        """Extract brightness values at joint coordinates from video."""
        print(f"Extracting brightness from video: {os.path.basename(video_path)}")

        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"❌ Failed to open video: {video_path}")
                return {}

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            print(f"   Video info: {total_frames} frames, {fps:.2f} fps")

            # Limit frames if specified
            max_frames = frame_limit if frame_limit else total_frames
            max_frames = min(max_frames, total_frames)

            # Initialize brightness storage
            joint_brightness = {joint: [] for joint in coordinates.keys()}

            frame_idx = 0
            while frame_idx < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                # Convert to grayscale for brightness calculation
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame_height, frame_width = gray_frame.shape

                # Extract brightness for each joint
                for joint_name, joint_coords in coordinates.items():
                    if frame_idx < len(joint_coords):
                        x, y = joint_coords[frame_idx]

                        # Ensure coordinates are within frame bounds
                        x = int(np.clip(x, 0, frame_width - 1))
                        y = int(np.clip(y, 0, frame_height - 1))

                        # Extract brightness in a small region around the joint
                        x_min = max(0, x - sampling_radius)
                        x_max = min(frame_width, x + sampling_radius + 1)
                        y_min = max(0, y - sampling_radius)
                        y_max = min(frame_height, y + sampling_radius + 1)

                        region = gray_frame[y_min:y_max, x_min:x_max]

                        if region.size > 0:
                            brightness = np.mean(region)
                        else:
                            brightness = gray_frame[y, x]  # Single pixel fallback

                        joint_brightness[joint_name].append(float(brightness))
                    else:
                        # No coordinate data for this frame
                        joint_brightness[joint_name].append(np.nan)

                frame_idx += 1

                # Progress update
                if frame_idx % 100 == 0:
                    print(f"   Processed {frame_idx}/{max_frames} frames")

            cap.release()

            print(
                f"✅ Extracted brightness for {len(joint_brightness)} joints over {frame_idx} frames"
            )
            return joint_brightness

        except Exception as e:
            print(f"❌ Failed to extract brightness: {e}")
            return {}

    def extract_brightness_for_joints(
        self,
        joint_names: List[str] = None,
        video_path: str = None,
        sampling_radius: int = 3,
        frame_limit: int = None,
        **gt_filter_kwargs,
    ) -> Dict[str, List[float]]:
        """Extract brightness for specified joints from ground truth and video."""
        print(f"Starting joint brightness extraction for joints: {joint_names}")

        try:
            # Get joint coordinates from ground truth
            coordinates = self.gt_loader.extract_joint_coordinates(
                joint_names, **gt_filter_kwargs
            )

            if not coordinates:
                print("❌ No joint coordinates available")
                return {}

            # Determine video path
            if video_path is None:
                video_files = self.get_video_files()
                if not video_files:
                    print("❌ No video files found")
                    return {}
                video_path = video_files[0]  # Use first video found
                print(f"Using video: {video_path}")

            # Extract brightness
            brightness_data = self.extract_brightness_at_coordinates(
                video_path, coordinates, sampling_radius, frame_limit
            )

            return brightness_data

        except Exception as e:
            print(f"❌ Failed to extract joint brightness: {e}")
            return {}

    def calculate_joint_brightness_statistics(
        self, brightness_data: Dict[str, List[float]]
    ) -> pd.DataFrame:
        """Calculate statistics for joint brightness values."""
        if not brightness_data:
            return pd.DataFrame()

        stats_data = []

        for joint_name, brightness_values in brightness_data.items():
            # Remove NaN values
            valid_values = [v for v in brightness_values if not np.isnan(v)]

            if not valid_values:
                continue

            stats = {
                "joint": joint_name,
                "frame_count": len(valid_values),
                "mean_brightness": np.mean(valid_values),
                "median_brightness": np.median(valid_values),
                "std_brightness": np.std(valid_values),
                "min_brightness": np.min(valid_values),
                "max_brightness": np.max(valid_values),
                "q25_brightness": np.percentile(valid_values, 25),
                "q75_brightness": np.percentile(valid_values, 75),
                "brightness_range": np.max(valid_values) - np.min(valid_values),
            }

            stats_data.append(stats)

        stats_df = pd.DataFrame(stats_data)

        if not stats_df.empty:
            # Round numeric columns
            numeric_columns = stats_df.select_dtypes(include=[np.number]).columns
            stats_df[numeric_columns] = stats_df[numeric_columns].round(3)

            print(f"✅ Calculated brightness statistics for {len(stats_df)} joints")

        return stats_df

    def export_brightness_data(
        self,
        brightness_data: Dict[str, List[float]],
        output_filename: str = None,
        include_stats: bool = True,
    ) -> str:
        """Export brightness data to CSV file."""
        try:
            if not brightness_data:
                print("❌ No brightness data to export")
                return ""

            # Prepare data for export
            max_frames = max(len(values) for values in brightness_data.values())

            export_data = []
            for frame_idx in range(max_frames):
                row = {"frame": frame_idx}
                for joint_name, values in brightness_data.items():
                    if frame_idx < len(values):
                        row[f"{joint_name}_brightness"] = values[frame_idx]
                    else:
                        row[f"{joint_name}_brightness"] = np.nan
                export_data.append(row)

            brightness_df = pd.DataFrame(export_data)

            # Generate filename
            if output_filename is None:
                joint_names = list(brightness_data.keys())
                joint_str = (
                    "_".join(joint_names[:3])
                    if len(joint_names) <= 3
                    else f"{len(joint_names)}_joints"
                )
                output_filename = (
                    f"joint_brightness_{self.dataset_name}_{joint_str}.csv"
                )

            # Save to config save folder
            output_path = os.path.join(self.config.save_folder, output_filename)
            os.makedirs(self.config.save_folder, exist_ok=True)

            brightness_df.to_csv(output_path, index=False)
            print(f"✅ Exported brightness data to: {output_path}")

            # Also export statistics if requested
            if include_stats:
                stats_df = self.calculate_joint_brightness_statistics(brightness_data)
                if not stats_df.empty:
                    stats_filename = output_filename.replace(".csv", "_stats.csv")
                    stats_path = os.path.join(self.config.save_folder, stats_filename)
                    stats_df.to_csv(stats_path, index=False)
                    print(f"✅ Exported brightness statistics to: {stats_path}")

            return output_path

        except Exception as e:
            print(f"❌ Failed to export brightness data: {e}")
            return ""

    def compare_joint_brightness(
        self, brightness_data: Dict[str, List[float]], reference_joint: str = None
    ) -> pd.DataFrame:
        """Compare brightness across different joints."""
        if not brightness_data:
            return pd.DataFrame()

        if reference_joint is None:
            reference_joint = list(brightness_data.keys())[0]

        if reference_joint not in brightness_data:
            print(f"❌ Reference joint {reference_joint} not found")
            return pd.DataFrame()

        print(f"Comparing joints against reference: {reference_joint}")

        reference_values = [
            v for v in brightness_data[reference_joint] if not np.isnan(v)
        ]
        if not reference_values:
            print("❌ No valid reference values")
            return pd.DataFrame()

        comparison_data = []

        for joint_name, values in brightness_data.items():
            if joint_name == reference_joint:
                continue

            valid_values = [v for v in values if not np.isnan(v)]
            if not valid_values:
                continue

            # Calculate comparison metrics
            mean_diff = np.mean(valid_values) - np.mean(reference_values)
            relative_brightness = np.mean(valid_values) / np.mean(reference_values)

            comparison = {
                "joint": joint_name,
                "reference_joint": reference_joint,
                "mean_brightness": np.mean(valid_values),
                "reference_mean": np.mean(reference_values),
                "brightness_difference": mean_diff,
                "brightness_ratio": relative_brightness,
                "brighter_than_reference": mean_diff > 0,
            }

            comparison_data.append(comparison)

        comparison_df = pd.DataFrame(comparison_data)

        if not comparison_df.empty:
            # Round numeric columns
            numeric_columns = comparison_df.select_dtypes(include=[np.number]).columns
            comparison_df[numeric_columns] = comparison_df[numeric_columns].round(4)

            print(f"✅ Generated brightness comparison for {len(comparison_df)} joints")

        return comparison_df


def main():
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Joint Brightness Extractor")
    parser.add_argument("dataset", help="Dataset name (e.g., 'humaneva', 'movi')")
    parser.add_argument(
        "--joints", nargs="*", help="Specific joints to analyze (default: all)"
    )
    parser.add_argument("--video", help="Path to specific video file")
    parser.add_argument("--subject", help="Filter ground truth by subject")
    parser.add_argument("--action", help="Filter ground truth by action")
    parser.add_argument("--camera", type=int, help="Filter ground truth by camera")
    parser.add_argument(
        "--radius", type=int, default=3, help="Sampling radius around joint"
    )
    parser.add_argument(
        "--frame-limit", type=int, help="Limit number of frames to process"
    )
    parser.add_argument(
        "--export", action="store_true", help="Export brightness data to CSV"
    )
    parser.add_argument("--filename", help="Custom output filename")
    parser.add_argument(
        "--stats-only", action="store_true", help="Show statistics only"
    )
    parser.add_argument("--compare", help="Compare joints against reference joint")

    args = parser.parse_args()

    try:
        extractor = JointBrightnessExtractor(args.dataset)

        # Set up ground truth filters
        gt_filters = {}
        if args.subject:
            gt_filters["subject"] = args.subject
        if args.action:
            gt_filters["action"] = args.action
        if args.camera is not None:
            gt_filters["camera"] = args.camera

        # Extract brightness data
        brightness_data = extractor.extract_brightness_for_joints(
            joint_names=args.joints,
            video_path=args.video,
            sampling_radius=args.radius,
            frame_limit=args.frame_limit,
            **gt_filters,
        )

        if not brightness_data:
            print("❌ No brightness data extracted")
            return

        # Show statistics
        if args.stats_only or not args.export:
            stats_df = extractor.calculate_joint_brightness_statistics(brightness_data)
            if not stats_df.empty:
                print("\n" + "=" * 60)
                print("JOINT BRIGHTNESS STATISTICS")
                print("=" * 60)
                print(stats_df.to_string(index=False))

        # Show comparison
        if args.compare:
            comparison_df = extractor.compare_joint_brightness(
                brightness_data, args.compare
            )
            if not comparison_df.empty:
                print("\n" + "=" * 60)
                print(f"BRIGHTNESS COMPARISON (vs {args.compare})")
                print("=" * 60)
                print(comparison_df.to_string(index=False))

        # Export data
        if args.export:
            extractor.export_brightness_data(
                brightness_data, args.filename, include_stats=True
            )

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

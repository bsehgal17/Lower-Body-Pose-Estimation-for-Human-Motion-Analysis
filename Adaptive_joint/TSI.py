"""
Temporal Stability Index (TSI) Computation Module

TSI measures the improvement in temporal stability when comparing filtered pose data to raw data.
It calculates the reduction in frame-to-frame jitter while accounting for the magnitude of motion.

TSI = (σ_raw - σ_filt) / σ_raw

Where:
- σ_raw: temporal std of frame-to-frame differences in raw poses
- σ_filt: temporal std of frame-to-frame differences in filtered poses
- Range: [0, 1] where higher = more stability improvement
- Negative values indicate over-smoothing
"""

import json
import os
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Tuple, List
import pandas as pd
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
from openpyxl.utils import get_column_letter

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TSICalculator:
    """Calculate Temporal Stability Index for pose estimation filtering."""

    def __init__(self, raw_path: str, filtered_path: str):
        """
        Initialize TSI Calculator.

        Args:
            raw_path: Path to directory containing raw pose JSON files
            filtered_path: Path to directory containing filtered pose JSON files
        """
        self.raw_path = Path(raw_path)
        self.filtered_path = Path(filtered_path)

        if not self.raw_path.exists():
            raise ValueError(f"Raw path does not exist: {raw_path}")
        if not self.filtered_path.exists():
            raise ValueError(f"Filtered path does not exist: {filtered_path}")

        logger.info(f"Raw path: {self.raw_path}")
        logger.info(f"Filtered path: {self.filtered_path}")

    def find_matching_files(self) -> List[Tuple[Path, Path]]:
        """
        Find matching JSON files in raw and filtered directories.

        Returns:
            List of tuples (raw_file_path, filtered_file_path)
        """
        raw_files = {f.stem: f for f in self.raw_path.glob("*.json")}
        filtered_files = {f.stem: f for f in self.filtered_path.glob("*.json")}

        common_stems = set(raw_files.keys()) & set(filtered_files.keys())

        if not common_stems:
            logger.warning(
                f"No matching files found between:\n"
                f"  Raw: {list(raw_files.keys())}\n"
                f"  Filtered: {list(filtered_files.keys())}"
            )

        matching_pairs = [
            (raw_files[stem], filtered_files[stem]) for stem in sorted(common_stems)
        ]

        logger.info(f"Found {len(matching_pairs)} matching file pairs")
        return matching_pairs

    @staticmethod
    def load_pose_data(json_path: Path) -> Dict:
        """
        Load pose data from JSON file.

        Args:
            json_path: Path to JSON file

        Returns:
            Dictionary containing pose data
        """
        try:
            with open(json_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading {json_path}: {e}")
            raise

    @staticmethod
    def extract_joint_trajectories(pose_data: Dict) -> Dict[int, np.ndarray]:
        """
        Extract joint trajectories from pose data.

        Expects pose_data to have structure:
        {
            "frames": [
                {"keypoints": [[x1, y1, c1], [x2, y2, c2], ...]},
                ...
            ]
        }
        or similar format with frames and keypoints.

        Args:
            pose_data: Loaded pose data dictionary

        Returns:
            Dictionary mapping joint_id to trajectory array (N_frames, 2)
        """
        trajectories = {}

        # Handle different pose data formats
        if "frames" in pose_data:
            frames = pose_data["frames"]
        elif "keypoints" in pose_data:
            frames = pose_data["keypoints"]
        else:
            logger.warning(f"Unexpected pose data format: {pose_data.keys()}")
            return trajectories

        if not frames:
            return trajectories

        # Extract trajectories for each joint
        first_frame = frames[0]
        if isinstance(first_frame, dict) and "keypoints" in first_frame:
            keypoints = first_frame["keypoints"]
        elif isinstance(first_frame, list):
            keypoints = first_frame
        else:
            return trajectories

        n_joints = len(keypoints)

        for joint_id in range(n_joints):
            trajectory = []
            for frame in frames:
                if isinstance(frame, dict) and "keypoints" in frame:
                    kp = frame["keypoints"][joint_id]
                else:
                    kp = frame[joint_id]

                # Extract x, y coordinates (first two values)
                if isinstance(kp, (list, tuple)) and len(kp) >= 2:
                    trajectory.append([kp[0], kp[1]])

            if trajectory:
                trajectories[joint_id] = np.array(trajectory)

        return trajectories

    @staticmethod
    def compute_temporal_std(trajectory: np.ndarray) -> float:
        """
        Compute temporal standard deviation from frame-to-frame differences.

        Args:
            trajectory: Array of shape (N_frames, 2) with x, y coordinates

        Returns:
            Standard deviation of frame-to-frame differences
        """
        if len(trajectory) < 2:
            return 0.0

        # Compute frame-to-frame differences (Δx)
        differences = np.diff(trajectory, axis=0)

        # Compute Euclidean distance of differences
        distances = np.linalg.norm(differences, axis=1)

        # Return standard deviation
        return float(np.std(distances))

    def compute_tsi_per_joint_per_video(
        self, raw_trajectory: np.ndarray, filtered_trajectory: np.ndarray
    ) -> float:
        """
        Compute TSI for a single joint in a single video.

        TSI = (σ_raw - σ_filt) / σ_raw

        Args:
            raw_trajectory: Raw pose trajectory (N_frames, 2)
            filtered_trajectory: Filtered pose trajectory (N_frames, 2)

        Returns:
            TSI value in range [-∞, 1]
        """
        sigma_raw = self.compute_temporal_std(raw_trajectory)
        sigma_filt = self.compute_temporal_std(filtered_trajectory)

        # Avoid division by zero
        if sigma_raw == 0:
            if sigma_filt == 0:
                return 0.0  # No motion in either
            else:
                return -np.inf  # Over-smoothing

        tsi = (sigma_raw - sigma_filt) / sigma_raw
        return float(tsi)

    def process_all_videos(
        self,
    ) -> Tuple[Dict[int, np.ndarray], Dict[int, Tuple[float, float]]]:
        """
        Process all matching video pairs and compute TSI.

        Returns:
            Tuple of:
            - tsi_matrix: Dict mapping joint_id to array of TSI values across videos
            - tsi_stats: Dict mapping joint_id to (mean, std) of TSI values
        """
        matching_pairs = self.find_matching_files()

        if not matching_pairs:
            raise ValueError("No matching file pairs found")

        # Dictionary to store TSI values: {joint_id: [tsi_video1, tsi_video2, ...]}
        tsi_per_joint = {}

        for video_idx, (raw_path, filtered_path) in enumerate(matching_pairs):
            logger.info(
                f"Processing video {video_idx + 1}/{len(matching_pairs)}: {raw_path.stem}"
            )

            # Load pose data
            raw_data = self.load_pose_data(raw_path)
            filtered_data = self.load_pose_data(filtered_path)

            # Extract trajectories
            raw_traj = self.extract_joint_trajectories(raw_data)
            filt_traj = self.extract_joint_trajectories(filtered_data)

            if not raw_traj or not filt_traj:
                logger.warning(f"Skipping {raw_path.stem}: empty trajectories")
                continue

            # Compute TSI for each joint
            for joint_id in raw_traj.keys():
                if joint_id not in filt_traj:
                    logger.warning(
                        f"Joint {joint_id} missing in filtered data for {raw_path.stem}"
                    )
                    continue

                tsi_value = self.compute_tsi_per_joint_per_video(
                    raw_traj[joint_id], filt_traj[joint_id]
                )

                if joint_id not in tsi_per_joint:
                    tsi_per_joint[joint_id] = []

                tsi_per_joint[joint_id].append(tsi_value)

        # Aggregate TSI across videos
        tsi_stats = {}
        tsi_matrix = {}

        for joint_id, tsi_values in sorted(tsi_per_joint.items()):
            tsi_array = np.array(tsi_values)
            tsi_matrix[joint_id] = tsi_array

            mean_tsi = float(np.mean(tsi_array))
            std_tsi = float(np.std(tsi_array))

            tsi_stats[joint_id] = (mean_tsi, std_tsi)

            # Check for over-smoothing
            negative_count = np.sum(tsi_array < 0)
            if negative_count > 0:
                logger.warning(
                    f"Joint {joint_id}: {negative_count}/{len(tsi_array)} "
                    f"videos show negative TSI (over-smoothing)"
                )

        return tsi_matrix, tsi_stats

    def generate_report(
        self,
        tsi_matrix: Dict[int, np.ndarray],
        tsi_stats: Dict[int, Tuple[float, float]],
    ) -> pd.DataFrame:
        """
        Generate a detailed TSI report.

        Args:
            tsi_matrix: TSI values per joint per video
            tsi_stats: Aggregated TSI statistics per joint

        Returns:
            DataFrame with TSI report
        """
        report_data = []

        for joint_id in sorted(tsi_stats.keys()):
            mean_tsi, std_tsi = tsi_stats[joint_id]
            tsi_values = tsi_matrix[joint_id]

            negative_tsi_count = np.sum(tsi_values < 0)
            max_tsi = float(np.max(tsi_values))
            min_tsi = float(np.min(tsi_values))

            report_data.append(
                {
                    "Joint_ID": joint_id,
                    "Mean_TSI": mean_tsi,
                    "Std_TSI": std_tsi,
                    "Min_TSI": min_tsi,
                    "Max_TSI": max_tsi,
                    "N_Negative": negative_tsi_count,
                    "N_Videos": len(tsi_values),
                }
            )

        df = pd.DataFrame(report_data)
        return df

    def print_summary(
        self, tsi_stats: Dict[int, Tuple[float, float]], df_report: pd.DataFrame
    ):
        """Print summary statistics."""
        print("\n" + "=" * 80)
        print("TEMPORAL STABILITY INDEX (TSI) SUMMARY")
        print("=" * 80)
        print(f"\nTotal joints analyzed: {len(tsi_stats)}\n")
        print(df_report.to_string(index=False))
        print("\n" + "=" * 80)

        overall_mean = df_report["Mean_TSI"].mean()
        overall_std = df_report["Std_TSI"].mean()

        print(f"\nOVERALL METRICS:")
        print(f"  Mean TSI (across all joints): {overall_mean:.4f}")
        print(f"  Avg Std TSI (stability across videos): {overall_std:.4f}")
        print(f"\nINTERPRETATION:")
        print(f"  TSI > 0.5: Excellent noise reduction")
        print(f"  0 < TSI < 0.5: Moderate noise reduction")
        print(f"  TSI ≈ 0: Minimal smoothing effect")
        print(f"  TSI < 0: Over-smoothing (loss of important motion)")
        print("=" * 80 + "\n")

    def save_to_excel(
        self,
        tsi_matrix: Dict[int, np.ndarray],
        tsi_stats: Dict[int, Tuple[float, float]],
        video_names: List[str],
        tsi_per_video_joint: Dict[Tuple[int, str], float],
        output_path: str = "TSI_Results.xlsx",
    ):
        """
        Save TSI results to Excel file with multiple sheets.

        Sheets created:
        1. "TSI_Per_Video": Rows=Videos, Columns=Joints, Values=TSI per video
        2. "TSI_Summary": Aggregated TSI statistics across all videos
        3. "TSI_Detailed": (Video, Joint) pairs with TSI values and interpretation

        Args:
            tsi_matrix: TSI values per joint per video
            tsi_stats: Aggregated TSI statistics per joint
            video_names: List of video names
            tsi_per_video_joint: Dict mapping (joint_id, video_name) to TSI value
            output_path: Output Excel file path
        """
        output_path = Path(output_path)

        try:
            with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
                # Sheet 1: TSI Per Video (Rows=Videos, Columns=Joints)
                self._create_tsi_per_video_sheet(
                    writer, tsi_per_video_joint, video_names, tsi_matrix
                )

                # Sheet 2: TSI Summary Statistics
                self._create_tsi_summary_sheet(writer, tsi_stats, tsi_matrix)

                # Sheet 3: TSI Detailed Results
                self._create_tsi_detailed_sheet(
                    writer, tsi_per_video_joint, video_names, tsi_stats
                )

            logger.info(f"Excel file saved: {output_path}")
            print(f"\n✓ Excel file saved: {output_path}")

        except Exception as e:
            logger.error(f"Error saving Excel file: {e}")
            raise

    def _create_tsi_per_video_sheet(
        self,
        writer: pd.ExcelWriter,
        tsi_per_video_joint: Dict[Tuple[int, str], float],
        video_names: List[str],
        tsi_matrix: Dict[int, np.ndarray],
    ):
        """Create sheet with TSI values per video (rows) and joint (columns)."""
        # Build data structure
        data = {}
        joint_ids = sorted(tsi_matrix.keys())

        for video_name in video_names:
            data[video_name] = {}
            for joint_id in joint_ids:
                key = (joint_id, video_name)
                if key in tsi_per_video_joint:
                    data[video_name][f"Joint_{joint_id}"] = round(
                        tsi_per_video_joint[key], 4
                    )
                else:
                    data[video_name][f"Joint_{joint_id}"] = None

        df = pd.DataFrame(data).T
        df.index.name = "Video"

        df.to_excel(writer, sheet_name="TSI_Per_Video")
        self._format_worksheet(
            writer.book["TSI_Per_Video"],
            header_color="4472C4",
            freeze_panes=(1, 1),
        )
        logger.info(f"Created sheet: TSI_Per_Video with {len(df)} videos")

    def _create_tsi_summary_sheet(
        self,
        writer: pd.ExcelWriter,
        tsi_stats: Dict[int, Tuple[float, float]],
        tsi_matrix: Dict[int, np.ndarray],
    ):
        """Create summary sheet with aggregated TSI statistics."""
        summary_data = []

        for joint_id in sorted(tsi_stats.keys()):
            mean_tsi, std_tsi = tsi_stats[joint_id]
            tsi_values = tsi_matrix[joint_id]

            negative_count = int(np.sum(tsi_values < 0))
            max_tsi = float(np.max(tsi_values))
            min_tsi = float(np.min(tsi_values))
            median_tsi = float(np.median(tsi_values))

            # Interpretation
            if mean_tsi > 0.5:
                interpretation = "Excellent"
            elif mean_tsi > 0.0:
                interpretation = "Moderate"
            elif mean_tsi >= -0.1:
                interpretation = "Minimal"
            else:
                interpretation = "Over-smoothing"

            summary_data.append(
                {
                    "Joint_ID": joint_id,
                    "Mean_TSI": round(mean_tsi, 4),
                    "Std_TSI": round(std_tsi, 4),
                    "Min_TSI": round(min_tsi, 4),
                    "Max_TSI": round(max_tsi, 4),
                    "Median_TSI": round(median_tsi, 4),
                    "N_Negative": negative_count,
                    "N_Videos": len(tsi_values),
                    "Interpretation": interpretation,
                }
            )

        df_summary = pd.DataFrame(summary_data)
        df_summary.to_excel(writer, sheet_name="TSI_Summary", index=False)
        self._format_worksheet(
            writer.book["TSI_Summary"], header_color="70AD47", freeze_panes=(1, 0)
        )
        logger.info(f"Created sheet: TSI_Summary with {len(df_summary)} joints")

    def _create_tsi_detailed_sheet(
        self,
        writer: pd.ExcelWriter,
        tsi_per_video_joint: Dict[Tuple[int, str], float],
        video_names: List[str],
        tsi_stats: Dict[int, Tuple[float, float]],
    ):
        """Create detailed sheet with all (video, joint) TSI values."""
        detailed_data = []

        for joint_id in sorted(tsi_stats.keys()):
            for video_name in video_names:
                key = (joint_id, video_name)
                if key in tsi_per_video_joint:
                    tsi_value = tsi_per_video_joint[key]

                    # Interpretation
                    if tsi_value > 0.5:
                        interp = "Excellent"
                    elif tsi_value > 0.0:
                        interp = "Moderate"
                    elif tsi_value >= -0.1:
                        interp = "Minimal"
                    else:
                        interp = "Over-smoothing"

                    detailed_data.append(
                        {
                            "Joint_ID": joint_id,
                            "Video": video_name,
                            "TSI": round(tsi_value, 4),
                            "Interpretation": interp,
                        }
                    )

        df_detailed = pd.DataFrame(detailed_data)
        df_detailed.to_excel(writer, sheet_name="TSI_Detailed", index=False)
        self._format_worksheet(
            writer.book["TSI_Detailed"], header_color="FFC7CE", freeze_panes=(1, 0)
        )
        logger.info(
            f"Created sheet: TSI_Detailed with {len(df_detailed)} video-joint pairs"
        )

    @staticmethod
    def _format_worksheet(
        worksheet, header_color: str = "4472C4", freeze_panes: Tuple = (1, 0)
    ):
        """Apply formatting to worksheet."""
        # Header styling
        header_fill = PatternFill(
            start_color=header_color, end_color=header_color, fill_type="solid"
        )
        header_font = Font(bold=True, color="FFFFFF")

        for cell in worksheet[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center", vertical="center")

        # Column width auto-adjust
        for column in worksheet.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 50)
            worksheet.column_dimensions[column_letter].width = adjusted_width

        # Apply borders
        thin_border = Border(
            left=Side(style="thin"),
            right=Side(style="thin"),
            top=Side(style="thin"),
            bottom=Side(style="thin"),
        )
        for row in worksheet.iter_rows(
            min_row=1,
            max_row=worksheet.max_row,
            min_col=1,
            max_col=worksheet.max_column,
        ):
            for cell in row:
                cell.border = thin_border

        # Freeze panes
        if freeze_panes != (0, 0):
            worksheet.freeze_panes = (
                f"{freeze_panes[0] + 1}{get_column_letter(freeze_panes[1] + 1)}"
            )


def main():
    """Main execution function."""

    # TODO: Update these paths to your actual data locations
    raw_poses_path = r"C:\path\to\raw\poses"  # Path to raw JSON files
    filtered_poses_path = r"C:\path\to\filtered\poses"  # Path to filtered JSON files

    # Initialize calculator
    calculator = TSICalculator(raw_poses_path, filtered_poses_path)

    # Process all videos
    tsi_matrix, tsi_stats, video_names, tsi_per_video_joint = (
        calculator.process_all_videos()
    )

    # Generate report
    df_report = calculator.generate_report(tsi_matrix, tsi_stats)

    # Print summary
    calculator.print_summary(tsi_stats, df_report)

    # Save to Excel
    calculator.save_to_excel(
        tsi_matrix, tsi_stats, video_names, tsi_per_video_joint, "TSI_Results.xlsx"
    )

    # Optionally save report as CSV
    # df_report.to_csv("tsi_report.csv", index=False)
    # logger.info("Report saved to tsi_report.csv")


if __name__ == "__main__":
    main()

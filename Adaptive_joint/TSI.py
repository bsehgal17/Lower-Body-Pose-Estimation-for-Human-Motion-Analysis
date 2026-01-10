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
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Tuple, List
import pandas as pd
from openpyxl.utils import get_column_letter, column_index_from_string
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TSICalculator:
    """Calculate Temporal Stability Index for pose estimation filtering."""

    # Only include these joints
    ALLOWED_JOINTS = {
        1, 2, 3, 4, 5, 6,      # hips, knees, ankles
        11, 12, 13, 14, 15, 16  # shoulders, elbows, wrists
    }

    def __init__(self, raw_path: str, filtered_path: str):
        self.raw_path = Path(raw_path)
        self.filtered_path = Path(filtered_path)

        if not self.raw_path.exists():
            raise ValueError(f"Raw path does not exist: {raw_path}")
        if not self.filtered_path.exists():
            raise ValueError(f"Filtered path does not exist: {filtered_path}")

        logger.info(f"Raw path: {self.raw_path}")
        logger.info(f"Filtered path: {self.filtered_path}")

    def extract_video_id(self, json_path: Path) -> Tuple[str, str, str]:
        """Extract (subject, motion, camera) from path."""
        subject = motion = camera = None
        for part in json_path.parts:
            if re.fullmatch(r"S\d+", part):
                subject = part
            m = re.fullmatch(r"(.+)_\((C\d+)\)", part)
            if m:
                motion = m.group(1)
                camera = m.group(2)

        if subject is None or motion is None or camera is None:
            raise ValueError(
                f"Could not parse video ID from path: {json_path}")

        return subject, motion, camera

    def find_matching_files(self) -> List[Tuple[Path, Path]]:
        """Match raw and filtered JSON files using (Subject, Motion, Camera)."""
        raw_map = {}
        for f in self.raw_path.rglob("*.json"):
            try:
                key = self.extract_video_id(f)
                raw_map[key] = f
            except ValueError as e:
                logger.warning(e)

        filt_map = {}
        for f in self.filtered_path.rglob("*.json"):
            try:
                key = self.extract_video_id(f)
                filt_map[key] = f
            except ValueError as e:
                logger.warning(e)

        common_keys = set(raw_map.keys()) & set(filt_map.keys())
        if not common_keys:
            logger.warning("No matching (Subject, Motion, Camera) found")

        matching_pairs = [(raw_map[k], filt_map[k])
                          for k in sorted(common_keys)]
        logger.info(f"Found {len(matching_pairs)} matching video pairs")
        return matching_pairs

    @staticmethod
    def load_pose_data(json_path: Path) -> Dict:
        try:
            with open(json_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading {json_path}: {e}")
            raise

    @staticmethod
    def extract_joint_trajectories(pose_data: Dict, allowed_joints: set) -> Dict[int, np.ndarray]:
        """Extract only selected joint trajectories from RTMW-style nested pose data."""
        trajectories = {}

        if "persons" not in pose_data or not pose_data["persons"]:
            logger.warning("No persons found in pose data")
            return trajectories

        # Primary person: longest track
        person = max(pose_data["persons"],
                     key=lambda p: len(p.get("poses", [])))
        poses = person.get("poses", [])
        if len(poses) < 2:
            logger.warning("Selected person has fewer than 2 frames")
            return trajectories

        first_pose = poses[0]
        if "keypoints" not in first_pose or not first_pose["keypoints"]:
            logger.warning("Pose does not contain keypoints")
            return trajectories

        n_joints = len(first_pose["keypoints"][0])

        for joint_id in range(n_joints):
            if joint_id not in allowed_joints:
                continue  # Skip joints not in your selected list

            traj = []
            for frame in poses:
                kps = frame.get("keypoints", [])
                if not kps or joint_id >= len(kps[0]):
                    continue
                kp = kps[0][joint_id]
                if isinstance(kp, (list, tuple)) and len(kp) >= 2:
                    traj.append([kp[0], kp[1]])
            if len(traj) >= 2:
                trajectories[joint_id] = np.asarray(traj, dtype=np.float32)

        logger.info(f"Extracted {len(trajectories)} joints (filtered)")
        return trajectories

    @staticmethod
    def compute_temporal_std(trajectory: np.ndarray) -> float:
        if len(trajectory) < 2:
            return 0.0
        differences = np.diff(trajectory, axis=0)
        distances = np.linalg.norm(differences, axis=1)
        return float(np.std(distances))

    def compute_tsi_per_joint_per_video(self, raw_trajectory: np.ndarray, filtered_trajectory: np.ndarray) -> float:
        sigma_raw = self.compute_temporal_std(raw_trajectory)
        sigma_filt = self.compute_temporal_std(filtered_trajectory)
        if sigma_raw == 0:
            return 0.0 if sigma_filt == 0 else -np.inf
        return float((sigma_raw - sigma_filt) / sigma_raw)

    def process_all_videos(self) -> Tuple[
        Dict[int, np.ndarray],
        Dict[int, Tuple[float, float]],
        List[str],
        Dict[Tuple[int, str], float],
        Dict[str, float]
    ]:
        """Process all videos and compute TSI for selected joints."""
        matching_pairs = self.find_matching_files()
        if not matching_pairs:
            raise ValueError("No matching file pairs found")

        tsi_per_joint = {}
        tsi_per_video_joint = {}
        tsi_per_video_mean = {}
        video_names = []

        for video_idx, (raw_path, filtered_path) in enumerate(matching_pairs):
            try:
                subject, motion, camera = self.extract_video_id(raw_path)
                video_name = f"{subject}_{motion}_{camera}"
            except Exception:
                video_name = raw_path.stem

            video_names.append(video_name)
            logger.info(
                f"Processing video {video_idx + 1}/{len(matching_pairs)}: {video_name}")

            raw_data = self.load_pose_data(raw_path)
            filtered_data = self.load_pose_data(filtered_path)

            raw_traj = self.extract_joint_trajectories(
                raw_data, allowed_joints=self.ALLOWED_JOINTS)
            filt_traj = self.extract_joint_trajectories(
                filtered_data, allowed_joints=self.ALLOWED_JOINTS)

            if not raw_traj or not filt_traj:
                logger.warning(f"Skipping {video_name}: empty trajectories")
                continue

            tsi_values_for_video = []

            for joint_id in raw_traj.keys():
                if joint_id not in self.ALLOWED_JOINTS:
                    continue  # Skip unwanted joints

                if joint_id not in filt_traj:
                    logger.warning(
                        f"Joint {joint_id} missing in filtered data for {video_name}")
                    continue

                tsi_value = self.compute_tsi_per_joint_per_video(
                    raw_traj[joint_id], filt_traj[joint_id])
                tsi_per_joint.setdefault(joint_id, []).append(tsi_value)
                tsi_per_video_joint[(joint_id, video_name)] = tsi_value
                tsi_values_for_video.append(tsi_value)

            # Mean TSI for video across selected joints
            tsi_per_video_mean[video_name] = float(
                np.mean(tsi_values_for_video)) if tsi_values_for_video else np.nan

        tsi_stats = {}
        tsi_matrix = {}
        for joint_id, tsi_values in tsi_per_joint.items():
            tsi_array = np.array(tsi_values)
            tsi_matrix[joint_id] = tsi_array
            tsi_stats[joint_id] = (
                float(np.mean(tsi_array)), float(np.std(tsi_array)))

            negative_count = np.sum(tsi_array < 0)
            if negative_count > 0:
                logger.warning(
                    f"Joint {joint_id}: {negative_count}/{len(tsi_array)} videos show negative TSI")

        return tsi_matrix, tsi_stats, video_names, tsi_per_video_joint, tsi_per_video_mean

    # ---------------- Reports ----------------
    def generate_report(self, tsi_matrix: Dict[int, np.ndarray], tsi_stats: Dict[int, Tuple[float, float]]) -> pd.DataFrame:
        report_data = []
        for joint_id in sorted(tsi_stats.keys()):
            mean_tsi, std_tsi = tsi_stats[joint_id]
            tsi_values = tsi_matrix[joint_id]
            negative_tsi_count = np.sum(tsi_values < 0)
            report_data.append({
                "Joint_ID": joint_id,
                "Mean_TSI": mean_tsi,
                "Std_TSI": std_tsi,
                "Min_TSI": float(np.min(tsi_values)),
                "Max_TSI": float(np.max(tsi_values)),
                "N_Negative": negative_tsi_count,
                "N_Videos": len(tsi_values),
            })
        return pd.DataFrame(report_data)

    def print_summary(self, tsi_stats: Dict[int, Tuple[float, float]], df_report: pd.DataFrame):
        print("\n" + "="*80)
        print("TEMPORAL STABILITY INDEX (TSI) SUMMARY")
        print("="*80)
        print(f"\nTotal joints analyzed: {len(tsi_stats)}\n")
        print(df_report.to_string(index=False))
        print("\n" + "="*80)
        overall_mean = df_report["Mean_TSI"].mean()
        overall_std = df_report["Std_TSI"].mean()
        print(f"\nOVERALL METRICS:")
        print(f"  Mean TSI (all joints): {overall_mean:.4f}")
        print(f"  Avg Std TSI: {overall_std:.4f}")
        print("="*80 + "\n")

    # ---------------- Excel Export ----------------
    def save_to_excel(self, tsi_matrix, tsi_stats, video_names, tsi_per_video_joint, tsi_per_video_mean, output_path="TSI_Results.xlsx"):
        output_path = Path(output_path)
        try:
            with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
                self._create_tsi_per_video_sheet(
                    writer, tsi_per_video_joint, video_names, tsi_matrix)
                self._create_tsi_summary_sheet(writer, tsi_stats, tsi_matrix)
                self._create_tsi_detailed_sheet(
                    writer, tsi_per_video_joint, video_names, tsi_stats)
                self._create_tsi_per_video_mean_sheet(
                    writer, tsi_per_video_mean)
            logger.info(f"Excel saved: {output_path}")
        except Exception as e:
            logger.error(f"Error saving Excel file: {e}")
            raise

    # ---- Excel sheet helpers ----
    def _create_tsi_per_video_sheet(self, writer, tsi_per_video_joint, video_names, tsi_matrix):
        data = {}
        joint_ids = sorted(tsi_matrix.keys())
        for video_name in video_names:
            data[video_name] = {}
            for joint_id in joint_ids:
                key = (joint_id, video_name)
                data[video_name][f"Joint_{joint_id}"] = round(
                    tsi_per_video_joint.get(key, np.nan), 4)
        df = pd.DataFrame(data).T
        df.index.name = "Video"
        df.to_excel(writer, sheet_name="TSI_Per_Video")
        self._format_worksheet(
            writer.book["TSI_Per_Video"], freeze_panes=(1, 1))
        logger.info(f"Created sheet: TSI_Per_Video")

    def _create_tsi_summary_sheet(self, writer, tsi_stats, tsi_matrix):
        summary_data = []
        for joint_id in sorted(tsi_stats.keys()):
            mean_tsi, std_tsi = tsi_stats[joint_id]
            tsi_values = tsi_matrix[joint_id]
            negative_count = int(np.sum(tsi_values < 0))
            summary_data.append({
                "Joint_ID": joint_id,
                "Mean_TSI": round(mean_tsi, 4),
                "Std_TSI": round(std_tsi, 4),
                "Min_TSI": round(float(np.min(tsi_values)), 4),
                "Max_TSI": round(float(np.max(tsi_values)), 4),
                "Median_TSI": round(float(np.median(tsi_values)), 4),
                "N_Negative": negative_count,
                "N_Videos": len(tsi_values),
            })
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_excel(writer, sheet_name="TSI_Summary", index=False)
        self._format_worksheet(writer.book["TSI_Summary"], freeze_panes=(1, 0))
        logger.info(f"Created sheet: TSI_Summary")

    def _create_tsi_detailed_sheet(self, writer, tsi_per_video_joint, video_names, tsi_stats):
        detailed_data = []
        for joint_id in sorted(tsi_stats.keys()):
            for video_name in video_names:
                key = (joint_id, video_name)
                if key in tsi_per_video_joint:
                    tsi_value = tsi_per_video_joint[key]
                    if tsi_value > 0.5:
                        interp = "Excellent"
                    elif tsi_value > 0.0:
                        interp = "Moderate"
                    elif tsi_value >= -0.1:
                        interp = "Minimal"
                    else:
                        interp = "Over-smoothing"
                    detailed_data.append({
                        "Joint_ID": joint_id,
                        "Video": video_name,
                        "TSI": round(tsi_value, 4),
                        "Interpretation": interp
                    })
        df_detailed = pd.DataFrame(detailed_data)
        df_detailed.to_excel(writer, sheet_name="TSI_Detailed", index=False)
        self._format_worksheet(
            writer.book["TSI_Detailed"], freeze_panes=(1, 0))
        logger.info(f"Created sheet: TSI_Detailed")

    def _create_tsi_per_video_mean_sheet(self, writer, tsi_per_video_mean):
        df_mean = pd.DataFrame(list(tsi_per_video_mean.items()), columns=[
                               "Video", "Mean_TSI_All_Joints"])
        df_mean.to_excel(writer, sheet_name="TSI_Per_Video_Mean", index=False)
        self._format_worksheet(
            writer.book["TSI_Per_Video_Mean"], freeze_panes=(1, 0))
        logger.info(f"Created sheet: TSI_Per_Video_Mean")

    @staticmethod
    def _format_worksheet(worksheet, header_color="4472C4", freeze_panes=(1, 0)):
        try:
            if freeze_panes != (0, 0):
                row = freeze_panes[0] + 1
                col_idx = freeze_panes[1] if isinstance(
                    freeze_panes[1], int) else column_index_from_string(freeze_panes[1])
                col_letter = get_column_letter(col_idx + 1)
                worksheet.freeze_panes = f"{col_letter}{row}"
        except Exception as e:
            logger.warning(f"Failed to set freeze_panes {freeze_panes}: {e}")


def main():
    raw_poses_path = r"/storageh100/Projects/Gaitly/bsehgal/pipeline_results/HumanEva/detect_RTMW"
    filtered_poses_path = r"/storageh100/Projects/Gaitly/bsehgal/pipeline_results/HumanEva/Butterworth_filter/filter_butterworth_18th_9hz"

    calculator = TSICalculator(raw_poses_path, filtered_poses_path)
    tsi_matrix, tsi_stats, video_names, tsi_per_video_joint, tsi_per_video_mean = calculator.process_all_videos()
    df_report = calculator.generate_report(tsi_matrix, tsi_stats)
    calculator.print_summary(tsi_stats, df_report)
    calculator.save_to_excel(
        tsi_matrix, tsi_stats, video_names, tsi_per_video_joint, tsi_per_video_mean,
        "/storage/Projects/Gaitly/bsehgal/lower_body_pose_est/pipeline_results/Adaptive_filt/TSI_Results.xlsx"
    )


if __name__ == "__main__":
    main()

import os
import pandas as pd
import pickle
from typing import Dict, List, Any
import numpy as np


class ResultAggregator:
    """
    Aggregates evaluation results (overall, joint-wise, and per-frame PCK/AP)
    and saves them to an Excel file or a pickle.
    """

    def __init__(self, output_path: str, save_as_pickle: bool = False):
        self.output_path = output_path
        self.overall_results: List[Dict[str, Any]] = []
        self.jointwise_results: List[Dict[str, Any]] = []
        self.per_frame_results: List[pd.DataFrame] = []
        # New lists to store AP results
        self.map_results: List[Dict[str, Any]] = []
        self.jointwise_ap_results: List[Dict[str, Any]] = []
        self.save_as_pickle = save_as_pickle

    def add_overall_result(self, metadata: Dict, value: float, threshold: float):
        self.overall_results.append({**metadata, f"PCK@{threshold}": value})

    def add_jointwise_result(
        self, metadata: Dict, joint_names: List[str], jointwise_scores: np.ndarray, threshold: float
    ):
        # The jointwise_scores are already averaged per joint
        if jointwise_scores.ndim == 1:
            jointwise_dict = {
                f"{j}_PCK@{threshold}": score for j, score in zip(joint_names, jointwise_scores)
            }
        # If the input is not yet averaged (from a jointwise PCK calc)
        else:
            jointwise_dict = {
                f"{j}_PCK@{threshold}": jointwise_scores[:, i].mean()
                for i, j in enumerate(joint_names)
            }
        self.jointwise_results.append({**metadata, **jointwise_dict})

    def add_per_frame_result(
        self, metadata: Dict, per_frame_scores: np.ndarray, threshold: float
    ):
        # Create a DataFrame for the per-frame scores of a single sample
        df = pd.DataFrame(per_frame_scores, columns=[f"PCK@{threshold}"])
        df['frame_idx'] = range(len(df))

        # Add metadata columns to the DataFrame
        for key, value in metadata.items():
            df[key] = value

        self.per_frame_results.append(df)

    def add_map_result(self, metadata: Dict, map_scores: Dict[str, Any]):
        """Adds overall mAP results to the aggregator."""
        # The mAP scores dictionary already contains the formatted keys
        self.map_results.append({**metadata, **map_scores})

    def add_jointwise_ap_result(self, metadata: Dict, jointwise_ap_scores: Dict[str, float]):
        """Adds joint-wise AP results to the aggregator."""
        # Prefix the scores with "AP@" for clarity in the dataframe
        prefixed_scores = {f"AP@{j}": s for j,
                           s in jointwise_ap_scores.items()}
        self.jointwise_ap_results.append({**metadata, **prefixed_scores})

    def save(self):
        """Saves all collected results into a single Excel file with multiple sheets or a single pickle file."""

        if self.save_as_pickle:
            # Save all results in a dictionary to a pickle file
            data_to_save = {
                "overall_pck": self.overall_results,
                "jointwise_pck": self.jointwise_results,
                "per_frame_pck": pd.concat(self.per_frame_results, ignore_index=True) if self.per_frame_results else pd.DataFrame(),
                "overall_map": self.map_results,
                "jointwise_ap": self.jointwise_ap_results,
            }
            with open(self.output_path, "wb") as f:
                pickle.dump(data_to_save, f)
            print(f"Saved results to {self.output_path}")
            return

        # Save to Excel
        with pd.ExcelWriter(self.output_path, engine='openpyxl') as writer:

            # Save Overall PCK Results
            if self.overall_results:
                df_overall_pck = pd.DataFrame(self.overall_results)
                df_overall_pck.to_excel(
                    writer, sheet_name="Overall PCK Scores", index=False)

            # Save Jointwise PCK Results
            if self.jointwise_results:
                df_jointwise_pck = pd.DataFrame(self.jointwise_results)
                df_jointwise_pck.to_excel(
                    writer, sheet_name="Jointwise PCK Scores", index=False)

            # Save Per-Frame PCK Results
            if self.per_frame_results:
                df_per_frame_pck = pd.concat(
                    self.per_frame_results, ignore_index=True)
                df_per_frame_pck.to_excel(
                    writer, sheet_name="Per-Frame PCK Scores", index=False)

            # Save Overall mAP Results
            if self.map_results:
                df_map = pd.DataFrame(self.map_results)
                df_map.to_excel(
                    writer, sheet_name="Overall mAP Scores", index=False)

            # Save Jointwise AP Results
            if self.jointwise_ap_results:
                df_jointwise_ap = pd.DataFrame(self.jointwise_ap_results)
                df_jointwise_ap.to_excel(
                    writer, sheet_name="Jointwise AP Scores", index=False)

        print(f"Saved results to {self.output_path}")

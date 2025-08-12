import os
import pandas as pd
import pickle
from typing import Dict, List, Any
import numpy as np


class ResultAggregator:
    """
    Aggregates evaluation results (overall, joint-wise, and per-frame)
    and saves them to an Excel file or a pickle.
    """

    def __init__(self, output_path: str, save_as_pickle: bool = False):
        self.output_path = output_path
        self.overall_results: List[Dict[str, Any]] = []
        self.jointwise_results: List[Dict[str, Any]] = []
        # New list to store per-frame DataFrames
        self.per_frame_results: List[pd.DataFrame] = []
        self.save_as_pickle = save_as_pickle

    def add_overall_result(self, metadata: Dict, value: float, threshold: float):
        self.overall_results.append({**metadata, f"PCK@{threshold}": value})

    def add_jointwise_result(
        self, metadata: Dict, joint_names: List[str], jointwise_scores: np.ndarray, threshold: float
    ):
        # Calculate mean for each joint across all frames
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

    def save(self):
        """Saves all collected results into a single Excel file with multiple sheets or a single pickle file."""

        if self.save_as_pickle:
            # Save all results in a dictionary to a pickle file
            data_to_save = {
                "overall": self.overall_results,
                "jointwise": self.jointwise_results,
                "per_frame": pd.concat(self.per_frame_results, ignore_index=True) if self.per_frame_results else pd.DataFrame()
            }
            with open(self.output_path, "wb") as f:
                pickle.dump(data_to_save, f)
            print(f"Saved results to {self.output_path}")
            return

        # Save to Excel
        with pd.ExcelWriter(self.output_path, engine='openpyxl') as writer:

            # Save Overall Results
            if self.overall_results:
                df_overall = pd.DataFrame(self.overall_results)
                df_overall.to_excel(
                    writer, sheet_name="Overall Scores", index=False)

            # Save Jointwise Results
            if self.jointwise_results:
                df_jointwise = pd.DataFrame(self.jointwise_results)
                df_jointwise.to_excel(
                    writer, sheet_name="Jointwise Scores", index=False)

            # Save Per-Frame Results
            if self.per_frame_results:
                df_per_frame = pd.concat(
                    self.per_frame_results, ignore_index=True)
                df_per_frame.to_excel(
                    writer, sheet_name="Per-Frame Scores", index=False)

        print(f"Saved results to {self.output_path}")

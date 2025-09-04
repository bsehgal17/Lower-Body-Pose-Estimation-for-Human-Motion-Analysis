import pandas as pd
import numpy as np
import re


class GroundTruthLoader:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df = self._load_and_prepare()

    def _load_and_prepare(self):
        df = pd.read_csv(self.csv_path)
        # Extract 'Jog 1', 'Walk 2', etc.
        df["action_group"] = df["Action"].str.extract(r"([a-zA-Z]+\s\d+)")
        return df

    def get_keypoints(
        self,
        subject,
        action_group,
        camera,
        chunk="chunk0",
        keypoint_columns=None,
    ):
        """
        Extracts keypoint data for the given filters.

        Parameters:
        - subject: Subject ID (e.g., "S1")
        - action_group: e.g., "Walk 1"
        - camera: int (0, 1, 2)
        - chunk: Optional chunk filter like "chunk0" (default)
        - keypoint_columns: List of columns to extract. If None, auto-detects x/y pairs.

        Returns:
        - keypoints: np.ndarray of shape (N, J, 2)
        """
        df_filtered = self.df[
            (self.df["Subject"] == subject)
            & (self.df["action_group"] == action_group)
            & (self.df["Camera"] == camera)
            & (self.df["Action"].str.contains(chunk))
        ]

        if df_filtered.empty:
            raise ValueError(
                f"No matching data found for Subject={subject}, Action={action_group}, Camera={camera}, Chunk={chunk}"
            )

        if keypoint_columns is None:
            keypoint_columns = [
                col for col in df_filtered.columns if re.match(r"[xy]\d+", col)
            ]

        keypoints = []
        for _, row in df_filtered.iterrows():
            row_kpts = row[keypoint_columns].values.astype(np.float64).reshape(-1, 2)
            keypoints.append(row_kpts)

        return np.array(keypoints)

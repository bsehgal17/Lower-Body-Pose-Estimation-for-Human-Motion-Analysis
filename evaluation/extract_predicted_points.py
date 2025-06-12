import numpy as np
import json
import os
import pickle


class PredictionExtractor:
    def __init__(self, file_path, file_format="json"):
        self.file_path = file_path
        self.file_format = file_format.lower()
        self.data = self._load_data()

    def _load_data(self):
        if self.file_format == "json":
            return self._load_json()
        elif self.file_format == "pkl":
            return self._load_pickle()
        else:
            raise ValueError(f"Unsupported file format: {self.file_format}")

    def _load_json(self):
        print(f"Loading JSON file from {self.file_path} ...")
        with open(self.file_path, "r") as f:
            data = json.load(f)
        print(f"Loaded {len(data)} frames from JSON.")
        return data

    def _load_pickle(self):
        print(f"Loading Pickle file from {self.file_path} ...")
        with open(self.file_path, "rb") as f:
            data = pickle.load(f)
        print(f"Loaded {len(data)} frames from Pickle.")
        return data

    def get_keypoint_array(self, frame_range=None):
        if self.file_format != "json":
            raise NotImplementedError(
                "Keypoint extraction only supported for JSON currently."
            )

        data = self.data

        total_frames = len(data)
        start, end = frame_range if frame_range else (0, total_frames)
        end = min(end, total_frames)

        keypoint_indices = {
            f"keypoint_{i}": i
            for i in range(len(data[0]["keypoints"][0]["keypoints"][0]))
        }

        pred_keypoints = []

        for i in range(start, end):
            frame_data = np.array(data[i]["keypoints"][0]["keypoints"][0])
            keypoints = [frame_data[idx] for idx in keypoint_indices.values()]
            pred_keypoints.append(np.array(keypoints))

        return np.array(pred_keypoints)

import json
import numpy as np
import os
from typing import Optional, List


class HumanSC3DGroundTruthLoader:
    """
    Ground truth loader for HumanSC3D dataset.

    Loads 2D keypoints from JSON files in the format:
    {
        "2d_keypoints": [
            [[x1, y1], [x2, y2], ..., [x25, y25]],  # Frame 1
            [[x1, y1], [x2, y2], ..., [x25, y25]],  # Frame 2
            ...
        ]
    }
    """

    def __init__(self, gt_path: str):
        """
        Initialize the ground truth loader.

        Args:
            gt_path: Path to the JSON file or directory containing JSON files
        """
        self.gt_path = gt_path
        if os.path.isfile(gt_path):
            self.data = self._load_single_file(gt_path)
        else:
            self.data = None

    def _load_single_file(self, json_path: str) -> dict:
        """Load a single JSON file."""
        with open(json_path, "r") as f:
            return json.load(f)

    def get_keypoints(
        self,
        subject: str = None,
        action: str = None,
        camera: str = None,
        joints_to_use: Optional[List[int]] = None,
    ) -> Optional[np.ndarray]:
        """
        Extract keypoint data for the given filters.

        Args:
            subject: Subject ID (e.g., 's06')
            action: Action ID (e.g., '170')
            camera: Camera ID (e.g., '50591643')
            joints_to_use: List of joint indices to extract. If None, returns all joints.

        Returns:
            numpy array of shape (n_frames, n_joints, 2) or None if no data found
        """
        if self.data is None:
            # If gt_path is a directory, construct the filename
            if os.path.isdir(self.gt_path):
                if not all([action, camera]):
                    print(
                        f"[Warning] Need action and camera to load from directory: {self.gt_path}"
                    )
                    return None

                filename = f"{action}_{camera}_2d.json"
                json_path = os.path.join(self.gt_path, filename)

                if not os.path.exists(json_path):
                    print(f"[Warning] Ground truth file not found: {json_path}")
                    return None

                try:
                    self.data = self._load_single_file(json_path)
                except Exception as e:
                    print(f"[Warning] Error loading {json_path}: {e}")
                    return None
            else:
                print(f"[Warning] Ground truth path does not exist: {self.gt_path}")
                return None

        if "2d_keypoints" not in self.data:
            print("[Warning] '2d_keypoints' key not found in ground truth data")
            return None

        keypoints = np.array(self.data["2d_keypoints"])

        # Filter joints if specified
        if joints_to_use is not None:
            if keypoints.shape[-1] >= 2:  # Ensure we have at least x,y coordinates
                keypoints = keypoints[:, joints_to_use, :]

        return keypoints

    def get_keypoints_from_path(
        self, json_path: str, joints_to_use: Optional[List[int]] = None
    ) -> Optional[np.ndarray]:
        """
        Load keypoints directly from a specific JSON file path.

        Args:
            json_path: Path to the JSON file
            joints_to_use: List of joint indices to extract

        Returns:
            numpy array of shape (n_frames, n_joints, 2) or None if error
        """
        try:
            data = self._load_single_file(json_path)

            if "2d_keypoints" not in data:
                print(f"[Warning] '2d_keypoints' key not found in {json_path}")
                return None

            keypoints = np.array(data["2d_keypoints"])

            # Filter joints if specified
            if joints_to_use is not None:
                if keypoints.shape[-1] >= 2:
                    keypoints = keypoints[:, joints_to_use, :]

            return keypoints

        except Exception as e:
            print(f"[Warning] Error loading keypoints from {json_path}: {e}")
            return None


def load_humansc3d_gt_keypoints(
    gt_path: str, joints_to_use: Optional[List[int]] = None
) -> Optional[np.ndarray]:
    """
    Convenience function to load HumanSC3D ground truth keypoints.

    Args:
        gt_path: Path to JSON file
        joints_to_use: List of joint indices to extract

    Returns:
        numpy array of shape (n_frames, n_joints, 2) or None if error
    """
    loader = HumanSC3DGroundTruthLoader(gt_path)
    return loader.get_keypoints_from_path(gt_path, joints_to_use)

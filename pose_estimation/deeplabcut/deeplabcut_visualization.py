import cv2
import numpy as np

class DeepLabCutVisualizer:
    def __init__(self, skeleton=None):
        self.skeleton = skeleton  # Optional list of joint index pairs

    def draw_keypoints(self, frame, keypoints_data, confidence_threshold=0.5):
        if isinstance(keypoints_data, np.ndarray) and keypoints_data.shape == ():
            keypoints_data = keypoints_data.item()  # Unwrap the object

        if not isinstance(keypoints_data, dict) or "bodyparts" not in keypoints_data:
            return frame  # Skip invalid frame

        bodyparts = keypoints_data["bodyparts"]
        if not isinstance(bodyparts, np.ndarray) or bodyparts.ndim != 3:
            return frame  # Invalid or unexpected shape

        for person_kpts in bodyparts:
            for x, y, conf in person_kpts:
                if conf > confidence_threshold:
                    cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)

            if self.skeleton:
                for i, j in self.skeleton:
                    if (
                        i < len(person_kpts)
                        and j < len(person_kpts)
                        and person_kpts[i][2] > confidence_threshold
                        and person_kpts[j][2] > confidence_threshold
                    ):
                        pt1 = tuple(map(int, person_kpts[i][:2]))
                        pt2 = tuple(map(int, person_kpts[j][:2]))
                        cv2.line(frame, pt1, pt2, (255, 0, 0), 1)

        return frame

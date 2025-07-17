import cv2
import numpy as np
import deeplabcut


class DeepLabCutVisualizer:
    def __init__(self, config_path, skeleton=None):
        self.config_path = config_path
        self.skeleton = skeleton  # Optional list of joint pairs

    def draw_keypoints(self, frame, keypoints, confidence_threshold=0.5):
        for point in keypoints:
            x, y, conf = point
            if conf > confidence_threshold:
                cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)

        if self.skeleton:
            for i, j in self.skeleton:
                if keypoints[i][2] > confidence_threshold and keypoints[j][2] > confidence_threshold:
                    pt1 = tuple(map(int, keypoints[i][:2]))
                    pt2 = tuple(map(int, keypoints[j][:2]))
                    cv2.line(frame, pt1, pt2, (255, 0, 0), 1)

        return frame

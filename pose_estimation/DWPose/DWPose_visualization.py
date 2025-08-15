import cv2
import numpy as np
import torch


class DWposeVisualizer:
    def __init__(self, skeleton=None):
        """
        Initialize the DWpose visualizer with optional skeleton connections.

        Args:
            skeleton (list): List of joint index pairs to connect with lines.
                           If None, uses default COCO whole-body connections.
        """
        self.skeleton = skeleton or self._get_default_skeleton()

    def _get_default_skeleton(self):
        """Default COCO whole-body skeleton connections for DWpose"""
        return [
            # Face (0-16)
            (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6),
            (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            # Body (17-22)
            (17, 18), (18, 19), (19, 20), (20, 21), (22, 23),
            # Arms (24-31)
            (24, 25), (25, 26), (26, 27), (27, 28), (28, 29),
            (24, 30), (30, 31), (31, 32), (32, 33), (33, 34),
            # Legs (35-42)
            (35, 36), (36, 37), (37, 38), (38, 39), (35, 40),
            (40, 41), (41, 42), (42, 43), (35, 44), (44, 45),
            (45, 46), (46, 47)
        ]

    def draw_keypoints(self, frame, keypoints_data, confidence_threshold=0.3):
        """
        Draw keypoints and skeleton on the frame.

        Args:
            frame (np.ndarray): Input image (BGR format)
            keypoints_data: DWpose output format (list of arrays with shape [N_joints, 3])
            confidence_threshold (float): Minimum confidence score to visualize a point

        Returns:
            np.ndarray: Frame with visualizations
        """
        if not isinstance(keypoints_data, list) or len(keypoints_data) == 0:
            return frame

        for person_kpts in keypoints_data:
            # Convert to numpy array if needed
            if isinstance(person_kpts, torch.Tensor):
                person_kpts = person_kpts.cpu().numpy()

            if not isinstance(person_kpts, np.ndarray) or person_kpts.ndim != 2:
                continue

            # Draw keypoints
            for x, y, conf in person_kpts:
                if conf > confidence_threshold:
                    cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)

            # Draw skeleton
            for i, j in self.skeleton:
                if (i < len(person_kpts) and j < len(person_kpts) and
                    person_kpts[i, 2] > confidence_threshold and
                        person_kpts[j, 2] > confidence_threshold):
                    pt1 = (int(person_kpts[i, 0]), int(person_kpts[i, 1]))
                    pt2 = (int(person_kpts[j, 0]), int(person_kpts[j, 1]))
                    cv2.line(frame, pt1, pt2, (255, 0, 0), 2)

        return frame

    def draw_detections(self, frame, bboxes, scores=None):
        """
        Draw detection bounding boxes on the frame.

        Args:
            frame (np.ndarray): Input image (BGR format)
            bboxes (np.ndarray): Array of bounding boxes in [x1, y1, x2, y2] format
            scores (np.ndarray): Optional array of confidence scores

        Returns:
            np.ndarray: Frame with bounding boxes drawn
        """
        if len(bboxes) == 0:
            return frame

        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = map(int, bbox[:4])
            color = (0, 255, 255)  # Yellow

            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw score if available
            if scores is not None and i < len(scores):
                score_text = f"{scores[i]:.2f}"
                cv2.putText(frame, score_text, (x1, y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return frame

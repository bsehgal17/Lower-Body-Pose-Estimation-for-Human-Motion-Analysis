import cv2
from utils.utils import combine_keypoints


class FrameProcessor:
    def __init__(self, detector, estimator, visualizer, config):
        self.detector = detector
        self.estimator = estimator
        self.visualizer = visualizer
        self.config = config

    def process_frame(self, frame, frame_idx, video_data):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        bboxes = self.detector.detect_humans(rgb)
        data_samples, pose_results = self.estimator.estimate_pose(frame, bboxes)
        combine_keypoints(pose_results, frame_idx, video_data, bboxes)
        return self.visualizer.visualize_pose(frame, data_samples)

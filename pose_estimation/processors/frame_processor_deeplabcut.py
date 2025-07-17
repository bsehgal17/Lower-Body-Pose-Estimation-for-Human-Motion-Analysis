import numpy as np


class FrameProcessorDLC:
    def __init__(self, detector, visualizer, pipeline_config):
        self.detector = detector
        self.visualizer = visualizer
        self.pipeline_config = pipeline_config

    def process_frame(self, frame, frame_idx, video_data):
        keypoints = self.detector.detect_and_estimate(frame)

        people = []
        for person_idx, kp in enumerate(keypoints):
            kp_filtered = kp.tolist()
            people.append({
                "id": f"idv_{person_idx}",
                "keypoints": kp_filtered
            })

        video_data.append({
            "frame_index": frame_idx,
            "people": people
        })

        if self.visualizer:
            frame = self.visualizer.draw_pose(frame, keypoints)

        return frame

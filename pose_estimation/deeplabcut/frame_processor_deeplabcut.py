class FrameProcessorDLC:
    def __init__(self, detector, visualizer, config):
        self.detector = detector
        self.visualizer = visualizer
        self.config = config

    def process_frame(self, frame, frame_idx, output_list):
        keypoints = self.detector.detect(frame)
        if self.visualizer:
            frame = self.visualizer.draw_keypoints(frame, keypoints)

        output_list.append(keypoints)
        return frame

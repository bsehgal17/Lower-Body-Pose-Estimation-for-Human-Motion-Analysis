import cv2


class VideoIO:
    def __init__(self, input_path, output_path, target_res=None):
        self.cap = cv2.VideoCapture(input_path)
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if target_res:
            w, h = target_res
        self.writer = cv2.VideoWriter(
            output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
        )

    def read(self):
        return self.cap.read()

    def write(self, frame):
        self.writer.write(frame)

    def release(self):
        self.cap.release()
        self.writer.release()

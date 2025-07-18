import numpy as np
import scipy.io as sio


class MoViGroundTruthLoader:
    def __init__(self, mat_path):
        self.data = sio.loadmat(
            mat_path, struct_as_record=False, squeeze_me=True)
        if "move" not in self.data:
            raise ValueError("Missing 'move' in MoVi file.")
        self.segments = self.data["move"]

    def get_keypoints(self, segment_index=0, joints_to_use=None):
        segment = self.segments[segment_index]
        joints = segment.jointsLocation_amass  # shape (N, J, 3)
        if joints_to_use is not None:
            joints = joints[:, joints_to_use, :]
        return joints[:, :, :2]  # return 2D joints

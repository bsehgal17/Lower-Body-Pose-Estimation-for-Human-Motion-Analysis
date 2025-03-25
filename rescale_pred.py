# utils/rescale_utils.py

import cv2


def get_video_resolution(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return width, height


def rescale_keypoints(pred_keypoints, scale_x, scale_y):
    rescaled = []
    for frame in pred_keypoints:
        rescaled_frame = [[x * scale_x, y * scale_y] for x, y in frame]
        rescaled.append(rescaled_frame)
    return rescaled

import cv2
import numpy as np


def apply_brightness_reduction(frame, noise_params):
    if not getattr(noise_params, "apply_brightness_reduction"):
        return frame

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)

    # Subtract brightness safely
    brightness_factor = noise_params.brightness_factor or 0
    v -= brightness_factor
    v = np.clip(v, 0, 255)

    hsv_mod = cv2.merge((h, s, v)).astype(np.uint8)
    return cv2.cvtColor(hsv_mod, cv2.COLOR_HSV2BGR)

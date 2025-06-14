import cv2


def apply_brightness_reduction(frame, noise_params):
    if not getattr(noise_params, "apply_brightness_reduction", False):
        return frame

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.subtract(v, noise_params.brightness_factor)
    hsv_mod = cv2.merge((h, s, v))
    return cv2.cvtColor(hsv_mod, cv2.COLOR_HSV2BGR)

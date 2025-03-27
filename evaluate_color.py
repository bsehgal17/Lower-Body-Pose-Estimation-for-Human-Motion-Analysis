import cv2
import numpy as np


def reduce_brightness_bgr(image, factor=40):
    """Reduce brightness in BGR (also works for RGB)."""
    return cv2.subtract(image, (factor, factor, factor))


def reduce_brightness_gray(image, factor=40):
    """Reduce brightness in grayscale."""
    return cv2.subtract(image, factor)


def reduce_brightness_hsv(image, factor=40):
    """Reduce brightness using the Value (V) channel in HSV."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.subtract(v, factor)
    hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def reduce_brightness_hls(image, factor=40):
    """Reduce brightness using the Lightness (L) channel in HLS."""
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    h, l, s = cv2.split(hls)
    l = cv2.subtract(l, factor)
    hls = cv2.merge((h, l, s))
    return cv2.cvtColor(hls, cv2.COLOR_HLS2BGR)


def reduce_brightness_ycrcb(image, factor=40):
    """Reduce brightness using the Y (Luma) channel in YCrCb."""
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y = cv2.subtract(y, factor)
    ycrcb = cv2.merge((y, cr, cb))
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def reduce_brightness_lab(image, factor=40):
    """Reduce brightness using the L (Lightness) channel in LAB."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(lab)
    l = cv2.subtract(l, factor)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)


def reduce_brightness_luv(image, factor=40):
    """Reduce brightness using the L (Lightness) channel in LUV."""
    luv = cv2.cvtColor(image, cv2.COLOR_BGR2Luv)
    l, u, v = cv2.split(luv)
    l = cv2.subtract(l, factor)
    luv = cv2.merge((l, u, v))
    return cv2.cvtColor(luv, cv2.COLOR_Luv2BGR)


def reduce_brightness_xyz(image, factor=40):
    """Reduce brightness using the Y (Luminance) channel in XYZ."""
    xyz = cv2.cvtColor(image, cv2.COLOR_BGR2XYZ)
    x, y, z = cv2.split(xyz)
    y = cv2.subtract(y, factor)
    xyz = cv2.merge((x, y, z))
    return cv2.cvtColor(xyz, cv2.COLOR_XYZ2BGR)


video_path = r"C:\Users\BhavyaSehgal\Downloads\bhavya_1st_sem\humaneva\HumanEva\HumanEva\S2\Image_Data\Walking_1_(C3).avi"
# Capture an image from the webcam or load an image
cap = cv2.VideoCapture(video_path)  # Use 0 for default webcam
ret, frame = cap.read()
cap.release()  # Release the webcam

if not ret:
    print("Error: Couldn't capture image")
    exit()

# Apply brightness reduction to all color spaces
images = {
    "org": frame,
    "BGR": reduce_brightness_bgr(frame),
    # "GRAY": reduce_brightness_gray(frame),
    "HSV": reduce_brightness_hsv(frame),
    # "HLS": reduce_brightness_hls(frame),
    # "YCrCb": reduce_brightness_ycrcb(frame),
    "LAB": reduce_brightness_lab(frame),
    # "LUV": reduce_brightness_luv(frame),
    # "XYZ": reduce_brightness_xyz(frame),
}

# Show results
for space, img in images.items():
    cv2.imshow(space, img)

cv2.waitKey(0)
cv2.destroyAllWindows()

import cv2
import numpy as np
from pathlib import Path


def apply_filter_then_clahe(input_path, output_dir):
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {input_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # Define CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # Filters to test (single filters only)
    filters = {
        "original": lambda x: x,
        "bilateral": lambda x: cv2.bilateralFilter(x, 9, 75, 75),
        "nlm": lambda x: cv2.fastNlMeansDenoisingColored(x, None, 10, 10, 7, 21),
        "gaussian": lambda x: cv2.GaussianBlur(x, (5, 5), 0),
        "laplacian_sharp": lambda x: cv2.addWeighted(
            x, 1.5, cv2.Laplacian(x, cv2.CV_64F).astype(np.uint8), -0.5, 0
        ),
        "cubic_resize": lambda x: cv2.resize(
            cv2.resize(x, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC),
            (x.shape[1], x.shape[0]),
            interpolation=cv2.INTER_CUBIC,
        ),
        "median": lambda x: cv2.medianBlur(x, 5),  # good for salt & pepper noise
        "gaussian_sharp": lambda x: cv2.addWeighted(
            x, 1.5, cv2.GaussianBlur(x, (5, 5), 0), -0.5, 0
        ),  # sharpening
    }

    # Create writers
    writers = {
        name: cv2.VideoWriter(
            str(output_dir / f"{name}_clahe.mp4"), fourcc, fps, (width, height)
        )
        for name in filters
    }

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        for name, func in filters.items():
            filtered = func(frame)

            # Apply CLAHE on V channel
            hsv = cv2.cvtColor(filtered, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            v_clahe = clahe.apply(v)
            hsv_clahe = cv2.merge((h, s, v_clahe))
            final = cv2.cvtColor(hsv_clahe, cv2.COLOR_HSV2BGR)

            writers[name].write(final)

    cap.release()
    for w in writers.values():
        w.release()

    print(f"Saved filtered + CLAHE videos to: {output_dir}")


# Example usage
if __name__ == "__main__":
    input_video = r"C:\Users\BhavyaSehgal\Downloads\bhavya_phd\dataset\HumanEvaFull\S3\Image_Data\Walking_2_(C3).avi"
    output_dir = r"C:\Users\BhavyaSehgal\Downloads\output_filters_clahe"
    apply_filter_then_clahe(input_video, output_dir)

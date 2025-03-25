import cv2
import numpy as np
import os
from utils import frame_generator  # Assuming your generator is defined in utils.py

# Load video using the frame generator
input_video_path = r"C:\Users\BhavyaSehgal\Downloads\bhavya_1st_sem\humaneva\HumanEva\HumanEva\S3\Image_Data\Walking_1_(C3).avi"
output_folder = r"C:\Users\BhavyaSehgal\Downloads\bhavya_1st_sem\humaneva\Human_eva_threshold\S3\Walking_1_(C3)_frames"

# Create output folder if not exists
os.makedirs(output_folder, exist_ok=True)


# Frame Processing Function
def process_frame(frame, frame_id):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive Thresholding to handle lighting variations
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    # Morphological operations to remove noise
    kernel = np.ones((3, 3), np.uint8)
    preprocessed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Save the preprocessed frame
    frame_path = os.path.join(output_folder, f"frame_{frame_id:04d}.png")
    cv2.imwrite(frame_path, preprocessed)

    # Display intermediate results
    cv2.imshow("Preprocessed Frame", preprocessed)
    if cv2.waitKey(1) & 0xFF == ord("q"):  # Press 'q' to exit
        return None

    return preprocessed


# Process each frame using the generator
frame_id = 0
for frame in frame_generator(input_video_path):
    processed_frame = process_frame(frame, frame_id)
    if processed_frame is None:
        break  # Exit if 'q' is pressed
    frame_id += 1

# Release resources
cv2.destroyAllWindows()
print(f"Processing complete! Preprocessed frames saved in: {output_folder}")

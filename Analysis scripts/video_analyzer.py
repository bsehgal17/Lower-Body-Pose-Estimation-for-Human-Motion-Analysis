import cv2
import numpy as np


class VideoAnalyzer:
    """
    Extracts brightness and contrast data from a video file.
    """

    def __init__(self, video_path):
        """
        Initializes the analyzer with a video file path.

        Args:
            video_path (str): Path to the video file.
        """
        self.video_path = video_path

    def get_brightness_data(self):
        """
        Reads a video file and calculates the average brightness (L-channel)
        for each frame.

        Returns:
            list: A list of average brightness values, one for each frame.
                  Returns an empty list if the video cannot be opened.
        """
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {self.video_path}")
            return []

        brightness_per_frame = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert frame to LAB color space
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

            # The L-channel (luminance) is at index 0.
            # Calculate the mean of the L-channel for the whole frame.
            avg_l_channel = np.mean(lab[:, :, 0])
            brightness_per_frame.append(avg_l_channel)

        cap.release()
        return brightness_per_frame

    def get_contrast_data(self):
        """
        Reads a video file and calculates the standard deviation of the
        brightness (L-channel) for each frame.

        Returns:
            list: A list of standard deviation values, one for each frame.
                  Returns an empty list if the video cannot be opened.
        """
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {self.video_path}")
            return []

        contrast_per_frame = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert frame to LAB color space
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

            # The L-channel (luminance) is at index 0.
            # Calculate the standard deviation of the L-channel for the whole frame.
            std_l_channel = np.std(lab[:, :, 0])
            contrast_per_frame.append(std_l_channel)

        cap.release()
        return contrast_per_frame

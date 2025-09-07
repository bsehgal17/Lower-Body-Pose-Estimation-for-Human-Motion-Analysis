"""
Image Filtering Module

This module provides various image filtering techniques that can be applied to frames
after enhancement techniques like CLAHE and gamma correction to further improve
image quality by reducing noise and smoothing details.
"""

import cv2
import numpy as np
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


class ImageFilter:
    """
    A class for applying various image filtering techniques to enhance image quality.

    This includes bilateral filtering, non-local means denoising, and Gaussian filtering.
    """

    @staticmethod
    def apply_bilateral_filter(
        frame: np.ndarray,
        d: int = 9,
        sigma_color: float = 75.0,
        sigma_space: float = 75.0,
    ) -> np.ndarray:
        """
        Apply bilateral filtering to reduce noise while preserving edges.

        Args:
            frame (np.ndarray): Input frame in BGR color space
            d (int): Diameter of neighborhood used during filtering (recommended: 5-9)
            sigma_color (float): Filter sigma in the color space (recommended: 10-150)
            sigma_space (float): Filter sigma in the coordinate space (recommended: 10-150)

        Returns:
            np.ndarray: Filtered frame
        """
        if frame is None or frame.size == 0:
            logger.warning("Empty or None frame provided to bilateral filter")
            return frame

        try:
            filtered_frame = cv2.bilateralFilter(frame, d, sigma_color, sigma_space)
            return filtered_frame
        except Exception as e:
            logger.error(f"Bilateral filtering failed: {e}")
            return frame

    @staticmethod
    def apply_non_local_means_filter(
        frame: np.ndarray,
        h: float = 10.0,
        template_window_size: int = 7,
        search_window_size: int = 21,
    ) -> np.ndarray:
        """
        Apply Non-Local Means denoising to reduce noise while preserving details.

        Args:
            frame (np.ndarray): Input frame in BGR color space
            h (float): Filter strength. Higher h removes more noise but also removes details (recommended: 3-10)
            template_window_size (int): Size of template patch (recommended: 7)
            search_window_size (int): Size of search window (recommended: 21)

        Returns:
            np.ndarray: Denoised frame
        """
        if frame is None or frame.size == 0:
            logger.warning("Empty or None frame provided to non-local means filter")
            return frame

        try:
            # Check if frame is grayscale or color
            if len(frame.shape) == 2:
                # Grayscale image
                filtered_frame = cv2.fastNlMeansDenoising(
                    frame, None, h, template_window_size, search_window_size
                )
            else:
                # Color image
                filtered_frame = cv2.fastNlMeansDenoisingColored(
                    frame, None, h, h, template_window_size, search_window_size
                )
            return filtered_frame
        except Exception as e:
            logger.error(f"Non-local means filtering failed: {e}")
            return frame

    @staticmethod
    def apply_gaussian_filter(
        frame: np.ndarray,
        kernel_size: Tuple[int, int] = (5, 5),
        sigma_x: float = 0.0,
        sigma_y: float = 0.0,
    ) -> np.ndarray:
        """
        Apply Gaussian filtering to smooth the image and reduce noise.

        Args:
            frame (np.ndarray): Input frame in BGR color space
            kernel_size (Tuple[int, int]): Size of the Gaussian kernel (must be odd)
            sigma_x (float): Gaussian kernel standard deviation in X direction
            sigma_y (float): Gaussian kernel standard deviation in Y direction
                           If sigma_y=0, it is set to sigma_x

        Returns:
            np.ndarray: Smoothed frame
        """
        if frame is None or frame.size == 0:
            logger.warning("Empty or None frame provided to Gaussian filter")
            return frame

        try:
            # Ensure kernel size values are odd
            ksize_x = kernel_size[0] if kernel_size[0] % 2 == 1 else kernel_size[0] + 1
            ksize_y = kernel_size[1] if kernel_size[1] % 2 == 1 else kernel_size[1] + 1

            filtered_frame = cv2.GaussianBlur(
                frame, (ksize_x, ksize_y), sigma_x, sigmaY=sigma_y
            )
            return filtered_frame
        except Exception as e:
            logger.error(f"Gaussian filtering failed: {e}")
            return frame

    @staticmethod
    def apply_median_filter(frame: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """
        Apply median filtering to remove salt-and-pepper noise while preserving edges.

        Args:
            frame (np.ndarray): Input frame in BGR color space
            kernel_size (int): Size of the median filter kernel (must be odd, typically 3, 5, or 7)

        Returns:
            np.ndarray: Filtered frame
        """
        if frame is None or frame.size == 0:
            logger.warning("Empty or None frame provided to median filter")
            return frame

        try:
            # Ensure kernel size is odd
            ksize = kernel_size if kernel_size % 2 == 1 else kernel_size + 1

            filtered_frame = cv2.medianBlur(frame, ksize)
            return filtered_frame
        except Exception as e:
            logger.error(f"Median filtering failed: {e}")
            return frame

    @staticmethod
    def apply_combined_filters(
        frame: np.ndarray, filters: list = None, filter_params: dict = None
    ) -> np.ndarray:
        """
        Apply multiple filters in sequence to a frame.

        Args:
            frame (np.ndarray): Input frame in BGR color space
            filters (list): List of filter names to apply in order
                          Options: 'bilateral', 'non_local_means', 'gaussian', 'median'
            filter_params (dict): Parameters for each filter

        Returns:
            np.ndarray: Filtered frame after applying all specified filters
        """
        if frame is None or frame.size == 0:
            logger.warning("Empty or None frame provided to combined filters")
            return frame

        if filters is None:
            filters = ["bilateral"]  # Default filter

        if filter_params is None:
            filter_params = {}

        filtered_frame = frame.copy()

        for filter_name in filters:
            if filter_name.lower() == "bilateral":
                params = filter_params.get("bilateral", {})
                filtered_frame = ImageFilter.apply_bilateral_filter(
                    filtered_frame, **params
                )

            elif filter_name.lower() == "non_local_means":
                params = filter_params.get("non_local_means", {})
                filtered_frame = ImageFilter.apply_non_local_means_filter(
                    filtered_frame, **params
                )

            elif filter_name.lower() == "gaussian":
                params = filter_params.get("gaussian", {})
                filtered_frame = ImageFilter.apply_gaussian_filter(
                    filtered_frame, **params
                )

            elif filter_name.lower() == "median":
                params = filter_params.get("median", {})
                filtered_frame = ImageFilter.apply_median_filter(
                    filtered_frame, **params
                )

            else:
                logger.warning(f"Unknown filter: {filter_name}")

        return filtered_frame


class FilteredCLAHEEnhancer:
    """
    Enhanced CLAHE class that applies image filtering after CLAHE enhancement.
    """

    def __init__(
        self,
        clip_limit: float,
        tile_grid_size: Tuple[int, int],
        filters: list = None,
        filter_params: dict = None,
    ):
        """
        Initialize the filtered CLAHE enhancer.

        Args:
            clip_limit (float): CLAHE clip limit
            tile_grid_size (Tuple[int, int]): CLAHE tile grid size
            filters (list): List of filters to apply after CLAHE
            filter_params (dict): Parameters for each filter
        """
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

        self.filters = filters if filters is not None else []
        self.filter_params = filter_params if filter_params is not None else {}

        logger.info(
            f"Filtered CLAHE enhancer initialized with clip_limit={clip_limit}, "
            f"tile_grid_size={tile_grid_size}, filters={self.filters}"
        )

    def enhance_frame(self, frame: np.ndarray, color_space: str = "HSV") -> np.ndarray:
        """
        Apply CLAHE enhancement followed by image filtering to a single frame.

        Args:
            frame (np.ndarray): Input frame in BGR color space
            color_space (str): Color space for CLAHE application (default: HSV)

        Returns:
            np.ndarray: Enhanced and filtered frame in BGR color space
        """
        if frame is None or frame.size == 0:
            logger.warning("Empty or None frame provided to enhance_frame")
            return frame

        # Step 1: Apply CLAHE enhancement
        enhanced_frame = self._apply_clahe(frame, color_space)

        # Step 2: Apply image filters
        if self.filters:
            filtered_frame = ImageFilter.apply_combined_filters(
                enhanced_frame, self.filters, self.filter_params
            )
            return filtered_frame

        return enhanced_frame

    def _apply_clahe(self, frame: np.ndarray, color_space: str) -> np.ndarray:
        """Apply CLAHE enhancement to frame."""
        # Handle grayscale images
        if len(frame.shape) == 2:
            return self.clahe.apply(frame)

        # Handle color images based on color space
        if color_space.upper() == "HSV":
            # Convert to HSV and apply CLAHE to V channel
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            h_channel, s_channel, v_channel = cv2.split(hsv)

            # Apply CLAHE to V channel (value/brightness)
            v_channel_clahe = self.clahe.apply(v_channel)

            # Merge channels back
            hsv_clahe = cv2.merge((h_channel, s_channel, v_channel_clahe))
            return cv2.cvtColor(hsv_clahe, cv2.COLOR_HSV2BGR)

        elif color_space.upper() == "LAB":
            # Convert to LAB and apply CLAHE to L channel
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)

            # Apply CLAHE to L channel (luminance)
            l_channel_clahe = self.clahe.apply(l_channel)

            # Merge channels back
            lab_clahe = cv2.merge((l_channel_clahe, a_channel, b_channel))
            return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

        elif color_space.upper() == "YUV":
            # Convert to YUV and apply CLAHE to Y channel
            yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
            y_channel, u_channel, v_channel = cv2.split(yuv)

            # Apply CLAHE to Y channel (luminance)
            y_channel_clahe = self.clahe.apply(y_channel)

            # Merge channels back
            yuv_clahe = cv2.merge((y_channel_clahe, u_channel, v_channel))
            return cv2.cvtColor(yuv_clahe, cv2.COLOR_YUV2BGR)

        elif color_space.upper() == "GRAY":
            # Convert to grayscale, apply CLAHE, then back to BGR
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_clahe = self.clahe.apply(gray)
            return cv2.cvtColor(gray_clahe, cv2.COLOR_GRAY2BGR)

        else:
            logger.error(f"Unsupported color space: {color_space}")
            return frame


class FilteredGammaEnhancer:
    """
    Enhanced Gamma correction class that applies image filtering after gamma enhancement.
    """

    def __init__(self, gamma: float, filters: list = None, filter_params: dict = None):
        """
        Initialize the filtered gamma enhancer.

        Args:
            gamma (float): Gamma correction value
            filters (list): List of filters to apply after gamma correction
            filter_params (dict): Parameters for each filter
        """
        self.gamma = gamma
        self.filters = filters if filters is not None else []
        self.filter_params = filter_params if filter_params is not None else {}

        # Pre-compute the lookup table for efficiency
        self._build_lookup_table()

        logger.info(
            f"Filtered Gamma enhancer initialized with gamma={gamma}, "
            f"filters={self.filters}"
        )

    def _build_lookup_table(self):
        """Build a lookup table for gamma correction to improve performance."""
        inv_gamma = 1.0 / self.gamma
        self.lookup_table = np.array(
            [((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]
        ).astype("uint8")

    def enhance_frame(self, frame: np.ndarray, color_space: str = "HSV") -> np.ndarray:
        """
        Apply gamma correction followed by image filtering to a single frame.

        Args:
            frame (np.ndarray): Input frame in BGR color space
            color_space (str): Color space for gamma application (default: HSV)

        Returns:
            np.ndarray: Enhanced and filtered frame in BGR color space
        """
        if frame is None or frame.size == 0:
            logger.warning("Empty or None frame provided to enhance_frame")
            return frame

        # Step 1: Apply gamma correction
        enhanced_frame = self._apply_gamma(frame, color_space)

        # Step 2: Apply image filters
        if self.filters:
            filtered_frame = ImageFilter.apply_combined_filters(
                enhanced_frame, self.filters, self.filter_params
            )
            return filtered_frame

        return enhanced_frame

    def _apply_gamma(self, frame: np.ndarray, color_space: str) -> np.ndarray:
        """Apply gamma correction to frame."""
        # Handle grayscale images
        if len(frame.shape) == 2:
            return cv2.LUT(frame, self.lookup_table)

        # Handle color images based on color space
        if color_space.upper() == "HSV":
            # Convert to HSV and apply gamma to V channel
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            h_channel, s_channel, v_channel = cv2.split(hsv)

            # Apply gamma correction to V channel (value/brightness)
            v_channel_gamma = cv2.LUT(v_channel, self.lookup_table)

            # Merge channels back
            hsv_gamma = cv2.merge((h_channel, s_channel, v_channel_gamma))
            return cv2.cvtColor(hsv_gamma, cv2.COLOR_HSV2BGR)

        elif color_space.upper() == "BGR":
            # Apply gamma correction to all channels
            return cv2.LUT(frame, self.lookup_table)

        elif color_space.upper() == "LAB":
            # Convert to LAB and apply gamma to L channel only
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)

            # Apply gamma correction to L channel (luminance)
            l_channel_gamma = cv2.LUT(l_channel, self.lookup_table)

            # Merge channels back
            lab_gamma = cv2.merge((l_channel_gamma, a_channel, b_channel))
            return cv2.cvtColor(lab_gamma, cv2.COLOR_LAB2BGR)

        elif color_space.upper() == "YUV":
            # Convert to YUV and apply gamma to Y channel
            yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
            y_channel, u_channel, v_channel = cv2.split(yuv)

            # Apply gamma correction to Y channel (luminance)
            y_channel_gamma = cv2.LUT(y_channel, self.lookup_table)

            # Merge channels back
            yuv_gamma = cv2.merge((y_channel_gamma, u_channel, v_channel))
            return cv2.cvtColor(yuv_gamma, cv2.COLOR_YUV2BGR)

        elif color_space.upper() == "GRAY":
            # Convert to grayscale, apply gamma, then back to BGR
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gamma_gray = cv2.LUT(gray, self.lookup_table)
            return cv2.cvtColor(gamma_gray, cv2.COLOR_GRAY2BGR)

        else:
            logger.warning(f"Unsupported color space: {color_space}. Using HSV.")
            return self._apply_gamma(frame, "HSV")

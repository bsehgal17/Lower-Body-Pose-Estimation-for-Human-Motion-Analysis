import numpy as np
from typing import Union
from scipy.signal import butter, filtfilt, savgol_filter, wiener
from scipy.ndimage import gaussian_filter1d, median_filter
from pykalman import KalmanFilter


# === Filter Implementations ===


def butterworth_filter(
    data: Union[list, np.ndarray], cutoff: float = 5, fs: float = 60.0, order: int = 10
) -> np.ndarray:
    """
    Applies a low-pass Butterworth filter to a 1D signal.

    Args:
        data: Input time series.
        cutoff: Cutoff frequency.
        fs: Sampling frequency.
        order: Order of the filter.

    Returns:
        Filtered 1D NumPy array.
    """
    if len(data) <= order:
        return np.array(data)
    b, a = butter(order, cutoff / (0.5 * fs), btype="low", analog=False)
    return filtfilt(b, a, data)


def gaussian_filter(data: Union[list, np.ndarray], sigma: float = 1) -> np.ndarray:
    """
    Applies a 1D Gaussian filter.

    Args:
        data: Input signal.
        sigma: Standard deviation of the Gaussian kernel.

    Returns:
        Smoothed signal.
    """
    return gaussian_filter1d(data, sigma=sigma)


def median_filter_1d(data: Union[list, np.ndarray], window_size: int = 5) -> np.ndarray:
    """
    Applies a median filter with a specified window size.

    Args:
        data: Input signal.
        window_size: Size of the median window (must be odd).

    Returns:
        Median-smoothed signal.
    """
    if len(data) == 0:
        return np.array([])
    return median_filter(data, size=window_size)


def initialize_kalman_filter(
    process_noise: float = 1.0, measurement_noise: float = 1.0
) -> KalmanFilter:
    """
    Initializes a Kalman Filter with a constant velocity model.

    Args:
        process_noise: Process noise covariance.
        measurement_noise: Measurement noise covariance.

    Returns:
        A KalmanFilter object.
    """
    return KalmanFilter(
        transition_matrices=np.array([[1, 1], [0, 1]]),
        observation_matrices=np.array([[1, 0]]),
        initial_state_mean=[0, 0],
        initial_state_covariance=np.eye(2),
        transition_covariance=process_noise * np.eye(2),
        observation_covariance=measurement_noise,
    )


def kalman_filter(
    data: Union[list, np.ndarray],
    process_noise: float = 1.0,
    measurement_noise: float = 1.0,
) -> np.ndarray:
    """
    Applies a Kalman filter to a 1D signal.

    Args:
        data: Input signal.
        process_noise: Process noise covariance.
        measurement_noise: Measurement noise covariance.

    Returns:
        Smoothed signal (position estimates).
    """
    if len(data) == 0:
        return np.array([])
    kf = initialize_kalman_filter(process_noise, measurement_noise)
    filtered, _ = kf.filter(data)
    return filtered[:, 0]


def kalman_rts_filter(
    data: Union[list, np.ndarray],
    process_noise: float = 1.0,
    measurement_noise: float = 1.0,
) -> np.ndarray:
    """
    Applies Kalman filtering followed by RTS smoothing.

    Args:
        data: Input signal.
        process_noise: Process noise covariance.
        measurement_noise: Measurement noise covariance.

    Returns:
        RTS-smoothed signal (position estimates).
    """
    if len(data) == 0:
        return np.array([])
    kf = initialize_kalman_filter(process_noise, measurement_noise)
    smoothed, _ = kf.smooth(data)
    return smoothed[:, 0]


def moving_average_filter(
    data: Union[list, np.ndarray], window_size: int = 5
) -> np.ndarray:
    """
    Applies a simple moving average filter to a 1D array.

    Args:
        data: Input signal.
        window_size: Window size for averaging.

    Returns:
        Smoothed signal.
    """
    num_frames = len(data)
    if num_frames < window_size:
        return np.array(data)

    half_window = window_size // 2
    smoothed_data = np.copy(data)

    for i in range(num_frames):
        start_idx = max(0, i - half_window)
        end_idx = min(num_frames, i + half_window + 1)
        smoothed_data[i] = np.mean(data[start_idx:end_idx])

    return smoothed_data


def savitzky_golay_filter(
    data: Union[list, np.ndarray], window_length: int = 11, polyorder: int = 3
) -> np.ndarray:
    """
    Applies Savitzky–Golay smoothing.

    Args:
        data: Input signal.
        window_length: Length of the filter window (must be odd and > polyorder).
        polyorder: Polynomial order to fit.

    Returns:
        Smoothed signal.
    """
    if len(data) < window_length:
        window_length = len(data) if len(data) % 2 == 1 else len(data) - 1
    if window_length < 3:
        return np.array(data)
    return savgol_filter(data, window_length=window_length, polyorder=polyorder)


def wiener_filter_1d(data: Union[list, np.ndarray], window_size: int = 3) -> np.ndarray:
    """
    Applies a Wiener filter to a 1D signal.

    Args:
        data: Input signal.
        window_size: Size of the filtering window.

    Returns:
        Denoised signal.
    """
    if len(data) < window_size:
        return np.array(data)
    return wiener(data, mysize=window_size)

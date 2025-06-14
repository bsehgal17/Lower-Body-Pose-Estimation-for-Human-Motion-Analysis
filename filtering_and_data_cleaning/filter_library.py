import numpy as np
from typing import Union
from scipy.signal import butter, filtfilt, savgol_filter, wiener
from scipy.ndimage import gaussian_filter1d, median_filter
from pykalman import KalmanFilter


class FilterLibrary:
    @staticmethod
    def butterworth(data, cutoff, fs, order):
        if len(data) <= order:
            return np.array(data)
        b, a = butter(order, cutoff / (0.5 * fs), btype="low", analog=False)
        return filtfilt(b, a, data)

    @staticmethod
    def gaussian(data, sigma):
        return gaussian_filter1d(data, sigma=sigma)

    @staticmethod
    def median(data, window_size):
        if len(data) == 0:
            return np.array([])
        return median_filter(data, size=window_size)

    @staticmethod
    def moving_average(data, window_size):
        num_frames = len(data)
        if num_frames < window_size:
            return np.array(data)
        half_window = window_size // 2
        return np.array(
            [
                np.mean(
                    data[max(0, i - half_window) : min(num_frames, i + half_window + 1)]
                )
                for i in range(num_frames)
            ]
        )

    @staticmethod
    def savitzky_golay(data, window_length, polyorder):
        if len(data) < window_length:
            window_length = len(data) if len(data) % 2 == 1 else len(data) - 1
        if window_length < 3:
            return np.array(data)
        return savgol_filter(data, window_length=window_length, polyorder=polyorder)

    @staticmethod
    def wiener(data, window_size):
        if len(data) < window_size:
            return np.array(data)
        return wiener(data, mysize=window_size)

    @staticmethod
    def kalman(data, process_noise, measurement_noise):
        if len(data) == 0:
            return np.array([])
        kf = FilterLibrary._init_kalman(process_noise, measurement_noise)
        filtered, _ = kf.filter(data)
        return filtered[:, 0]

    @staticmethod
    def kalman_rts(data, process_noise, measurement_noise):
        if len(data) == 0:
            return np.array([])
        kf = FilterLibrary._init_kalman(process_noise, measurement_noise)
        smoothed, _ = kf.smooth(data)
        return smoothed[:, 0]

    @staticmethod
    def _init_kalman(process_noise, measurement_noise):
        return KalmanFilter(
            transition_matrices=np.array([[1, 1], [0, 1]]),
            observation_matrices=np.array([[1, 0]]),
            initial_state_mean=[0, 0],
            initial_state_covariance=np.eye(2),
            transition_covariance=process_noise * np.eye(2),
            observation_covariance=measurement_noise,
        )

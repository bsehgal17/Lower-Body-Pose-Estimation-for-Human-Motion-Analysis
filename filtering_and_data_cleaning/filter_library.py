import numpy as np
from scipy.signal import butter, filtfilt, savgol_filter, wiener
from scipy.ndimage import gaussian_filter1d, median_filter
from pykalman import KalmanFilter
from scipy.interpolate import UnivariateSpline


class FilterLibrary:
    @staticmethod
    def butterworth(data, cutoff, fs, order):
        if len(data) <= order:
            return np.array(data)
        try:
            cutoff = float(cutoff)
            fs = float(fs)
            order = int(order)
        except Exception as e:
            raise ValueError(
                f"Invalid Butterworth parameters: cutoff={cutoff}, fs={fs}, order={order}") from e
        b, a = butter(order, cutoff / (0.5 * fs), btype="low", analog=False)
        return filtfilt(b, a, data)

    @staticmethod
    def gaussian(data, sigma):
        try:
            sigma = float(sigma)
        except Exception as e:
            raise ValueError(
                f"Invalid sigma for Gaussian filter: {sigma}") from e
        return gaussian_filter1d(data, sigma=sigma)

    @staticmethod
    def median(data, window_size):
        if len(data) == 0:
            return np.array([])
        try:
            window_size = int(window_size)
        except Exception as e:
            raise ValueError(
                f"Invalid window_size for median filter: {window_size}") from e
        return median_filter(data, size=window_size)

    @staticmethod
    def moving_average(data, window_size):
        num_frames = len(data)
        try:
            window_size = int(window_size)
        except Exception as e:
            raise ValueError(
                f"Invalid window_size for moving average: {window_size}") from e

        if num_frames < window_size:
            return np.array(data)
        half_window = window_size // 2
        return np.array([
            np.mean(data[max(0, i - half_window)
                    : min(num_frames, i + half_window + 1)])
            for i in range(num_frames)
        ])

    @staticmethod
    def savitzky_golay(data, window_length, polyorder):
        try:
            window_length = int(window_length)
            polyorder = int(polyorder)
        except Exception as e:
            raise ValueError(
                f"Invalid params for savgol filter: window_length={window_length}, polyorder={polyorder}") from e

        if len(data) < window_length:
            window_length = len(data) if len(data) % 2 == 1 else len(data) - 1
        if window_length < 3:
            return np.array(data)
        return savgol_filter(data, window_length=window_length, polyorder=polyorder)

    @staticmethod
    def wiener(data, window_size):
        try:
            window_size = int(window_size)
        except Exception as e:
            raise ValueError(
                f"Invalid window_size for Wiener filter: {window_size}") from e

        if len(data) < window_size:
            return np.array(data)
        return wiener(data, mysize=window_size)

    @staticmethod
    def kalman(data, process_noise, measurement_noise):
        if len(data) == 0:
            return np.array([])
        try:
            process_noise = float(process_noise)
            measurement_noise = float(measurement_noise)
        except Exception as e:
            raise ValueError(
                f"Invalid Kalman filter params: process_noise={process_noise}, measurement_noise={measurement_noise}") from e

        kf = FilterLibrary._init_kalman(process_noise, measurement_noise)
        filtered, _ = kf.filter(data)
        return filtered[:, 0]

    @staticmethod
    def gvcspl(data, smoothing_factor=None):
        """Generalized Variable Cutoff Spline Filter (Quintic Spline)."""
        if len(data) < 6:
            return np.array(data)
        x = np.arange(len(data))
        try:
            if smoothing_factor is not None:
                smoothing_factor = float(smoothing_factor)
        except Exception as e:
            raise ValueError(
                f"Invalid smoothing factor for GVCSPL: {smoothing_factor}") from e
        spline = UnivariateSpline(x, data, s=smoothing_factor, k=5)
        return spline(x)

    @staticmethod
    def extended_kalman(data, process_noise, measurement_noise):
        """Simplified 1D Extended Kalman Filter."""
        if len(data) == 0:
            return np.array([])
        try:
            process_noise = float(process_noise)
            measurement_noise = float(measurement_noise)
        except Exception as e:
            raise ValueError(
                f"Invalid EKF params: process_noise={process_noise}, measurement_noise={measurement_noise}") from e

        x = np.array([0.0])
        P = np.array([[1.0]])
        Q = np.array([[process_noise]])
        R = np.array([[measurement_noise]])
        H = np.array([[1.0]])
        F = np.array([[1.0]])
        results = []

        for z in data:
            x = F @ x
            P = F @ P @ F.T + Q
            y = z - H @ x
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            x = x + K @ y
            P = (np.eye(len(P)) - K @ H) @ P
            results.append(x[0])

        return np.array(results)

    @staticmethod
    def unscented_kalman(data, process_noise, measurement_noise):
        """Basic UKF implementation for 1D."""
        from filterpy.kalman import UnscentedKalmanFilter as UKF
        from filterpy.kalman import MerweScaledSigmaPoints

        if len(data) == 0:
            return np.array([])
        try:
            process_noise = float(process_noise)
            measurement_noise = float(measurement_noise)
        except Exception as e:
            raise ValueError(
                f"Invalid UKF params: process_noise={process_noise}, measurement_noise={measurement_noise}") from e

        def fx(x, dt): return x
        def hx(x): return x

        points = MerweScaledSigmaPoints(n=1, alpha=0.1, beta=2., kappa=0)
        ukf = UKF(dim_x=1, dim_z=1, dt=1.0, fx=fx, hx=hx, points=points)
        ukf.x = np.array([0.])
        ukf.P *= 1.
        ukf.Q *= process_noise
        ukf.R *= measurement_noise

        results = []
        for z in data:
            ukf.predict()
            ukf.update(z)
            results.append(ukf.x[0])
        return np.array(results)

    @staticmethod
    def _init_kalman(process_noise, measurement_noise):
        return KalmanFilter(
            transition_matrices=np.array([[1, 1], [0, 1]]),
            observation_matrices=np.array([[1, 0]]),
            initial_state_mean=[0, 0],
            initial_state_covariance=np.eye(2),
            transition_covariance=float(process_noise) * np.eye(2),
            observation_covariance=float(measurement_noise),
        )

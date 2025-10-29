import numpy as np
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
from scipy.signal import butter, sosfiltfilt, savgol_filter, wiener
from scipy.ndimage import gaussian_filter1d, median_filter
from pykalman import KalmanFilter
from scipy.interpolate import UnivariateSpline


class FilterLibrary:
    @staticmethod
    def butterworth(data, cutoff, fs, order):
        cutoff = float(cutoff)
        fs = float(fs)
        order = int(order)
        sos = butter(order, cutoff / (0.5 * fs), btype="low", output="sos")
        if len(data) <= 3 * (2 * order - 1):
            return np.array(data)
        return sosfiltfilt(sos, data)

    @staticmethod
    def gaussian(data, sigma):
        try:
            sigma = float(sigma)
        except Exception as e:
            raise ValueError(f"Invalid sigma for Gaussian filter: {sigma}") from e
        return gaussian_filter1d(data, sigma=sigma)

    @staticmethod
    def median(data, window_size):
        if len(data) == 0:
            return np.array([])
        try:
            window_size = int(window_size)
        except Exception as e:
            raise ValueError(
                f"Invalid window_size for median filter: {window_size}"
            ) from e
        return median_filter(data, size=window_size)

    @staticmethod
    def moving_average(data, window_size):
        num_frames = len(data)
        try:
            window_size = int(window_size)
        except Exception as e:
            raise ValueError(
                f"Invalid window_size for moving average: {window_size}"
            ) from e

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
        try:
            window_length = int(window_length)
            polyorder = int(polyorder)
        except Exception as e:
            raise ValueError(
                f"Invalid params for savgol filter: window_length={window_length}, polyorder={polyorder}"
            ) from e

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
                f"Invalid window_size for Wiener filter: {window_size}"
            ) from e

        if len(data) < window_size:
            return np.array(data)
        return wiener(data, mysize=window_size)

    @staticmethod
    def kalman(data, process_noise, measurement_noise, dt):
        """
        Apply 2D constant velocity Kalman Filter on gait joint data.

        Parameters:
        - data: n x 2 array of [x, y] positions
        - process_noise: expected acceleration variance (pixels^2/s^4)
        - measurement_noise: detector noise variance (pixels^2)
        - dt: time step between frames (s)

        Returns:
        - filtered positions: n x 2 array [x, y]
        """
        if len(data) == 0:
            return np.array([])

        try:
            process_noise = float(process_noise)
            measurement_noise = float(measurement_noise)
            dt = float(dt)
        except Exception as e:
            raise ValueError(
                f"Invalid Kalman filter params: process_noise={process_noise}, measurement_noise={measurement_noise}, dt={dt}"
            ) from e

        # State transition: [x, y, vx, vy]
        F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])

        # Measurement: only positions
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])

        # Process noise Q
        Q = process_noise * np.array(
            [
                [dt**4 / 4, 0, dt**3 / 2, 0],
                [0, dt**4 / 4, 0, dt**3 / 2],
                [dt**3 / 2, 0, dt**2, 0],
                [0, dt**3 / 2, 0, dt**2],
            ]
        )

        # Measurement noise R
        R = np.array([[measurement_noise, 0], [0, measurement_noise]])

        # Initial state: first measurement, zero velocity
        x0 = np.array([data[0, 0], data[0, 1], 0.0, 0.0])

        # Initial covariance
        P0 = np.diag([25**2, 25**2, 5**2, 5**2])

        # Initialize Kalman Filter
        kf = KalmanFilter(
            transition_matrices=F,
            observation_matrices=H,
            initial_state_mean=x0,
            initial_state_covariance=P0,
            transition_covariance=Q,
            observation_covariance=R,
        )

        # Apply filter
        filtered_state_means, _ = kf.filter(data)

        # Return only positions [x, y]
        return filtered_state_means[:, :2]

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
                f"Invalid smoothing factor for GVCSPL: {smoothing_factor}"
            ) from e
        spline = UnivariateSpline(x, data, s=smoothing_factor, k=5)
        return spline(x)

    @staticmethod
    def extended_kalman(data, process_noise, measurement_noise, dt):
        """
        2D Extended Kalman Filter for gait pose estimation (x, y) with velocity.
        - data: Nx2 array of joint positions (x, y) in pixels
        - process_noise: scalar (acceleration noise)
        - measurement_noise: scalar (position measurement noise)
        - dt: time step in seconds
        Returns filtered Nx2 array.
        """
        if len(data) == 0:
            return np.empty((0, 2))

        # State vector: [x, y, vx, vy]
        x = np.array([data[0, 0], data[0, 1], 0.0, 0.0])
        P = np.diag([25**2, 25**2, 5**2, 5**2])  # initial covariance

        # State transition matrix
        F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])

        # Measurement matrix
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])

        # Process noise Q (based on constant acceleration model)
        q = process_noise
        Q_block = np.array([[dt**4 / 4, dt**3 / 2], [dt**3 / 2, dt**2]]) * q**2
        Q = np.block([[Q_block, np.zeros((2, 2))], [np.zeros((2, 2)), Q_block]])

        # Measurement noise
        R = np.eye(2) * measurement_noise**2

        results = []
        for z in data:
            # Predict
            x = F @ x
            P = F @ P @ F.T + Q

            # Update
            y = z - H @ x
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            x = x + K @ y
            P = (np.eye(len(P)) - K @ H) @ P

            results.append(x[:2].copy())

        return np.array(results)

    @staticmethod
    def unscented_kalman(data, process_noise, measurement_noise, dt):
        """
        2D Unscented Kalman Filter for gait pose estimation (x, y) with velocity.
        - data: Nx2 array of joint positions (x, y) in pixels
        - process_noise: scalar (acceleration noise)
        - measurement_noise: scalar (position measurement noise)
        - dt: time step in seconds
        Returns filtered Nx2 array.
        """
        if len(data) == 0:
            return np.empty((0, 2))

        # UKF fx and hx functions for constant velocity
        def fx(x, dt):
            # x = [x, y, vx, vy]
            F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
            return F @ x

        def hx(x):
            # only measure positions
            return x[:2]

        points = MerweScaledSigmaPoints(n=4, alpha=1e-3, beta=2, kappa=0)
        ukf = UKF(dim_x=4, dim_z=2, dt=dt, fx=fx, hx=hx, points=points)

        # Initialize state with first measurement
        ukf.x = np.array([data[0, 0], data[0, 1], 0.0, 0.0])
        ukf.P = np.diag([25**2, 25**2, 5**2, 5**2])

        # Process & measurement noise
        q = process_noise
        Q_block = np.array([[dt**4 / 4, dt**3 / 2], [dt**3 / 2, dt**2]]) * q**2
        ukf.Q = np.block([[Q_block, np.zeros((2, 2))], [np.zeros((2, 2)), Q_block]])
        ukf.R = np.eye(2) * measurement_noise**2

        results = []
        for z in data:
            ukf.predict()
            ukf.update(z)
            results.append(ukf.x[:2].copy())

        return np.array(results)

    def fdf_filter(signal, cutoff, fs, order, window_type=None):
        """
        Frequency Domain Filter that mimics Butterworth behavior.

        Parameters:
        - signal: 1D NumPy array
        - cutoff: float, cutoff frequency in Hz
        - fs: float, sampling frequency in Hz
        - order: int, smoothness of the filter (higher = sharper roll-off)
        - window_type: Optional[str], type of window to apply ("hann", "hamming", None)

        Returns:
        - filtered: NumPy array of filtered signal
        """
        if len(signal) == 0:
            return np.array([])

        signal = np.asarray(signal, dtype=np.float64)

        # Frequencies
        n = len(signal)
        freqs = np.fft.rfftfreq(n, d=1 / fs)

        # FFT
        spectrum = np.fft.rfft(signal)

        # Butterworth-style low-pass mask
        mask = 1 / (1 + (freqs / cutoff) ** (2 * order))

        # Optional tapering window to reduce spectral leakage
        if window_type == "hann":
            taper = np.hanning(len(mask) * 2)[len(mask) :]
            mask *= taper
        elif window_type == "hamming":
            taper = np.hamming(len(mask) * 2)[len(mask) :]
            mask *= taper

        # Apply mask
        filtered_spectrum = spectrum * mask

        # Inverse FFT
        filtered_signal = np.fft.irfft(filtered_spectrum, n=n)

        return filtered_signal

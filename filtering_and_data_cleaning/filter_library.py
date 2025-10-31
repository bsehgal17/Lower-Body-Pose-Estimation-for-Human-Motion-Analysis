import numpy as np
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
from scipy.signal import butter, sosfiltfilt, savgol_filter, wiener
from scipy.ndimage import gaussian_filter1d, median_filter
from pykalman import KalmanFilter
from scipy.interpolate import UnivariateSpline
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints


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
                    data[max(0, i - half_window)
                             : min(num_frames, i + half_window + 1)]
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
        """1D Basic Kalman Filter with position & velocity."""
        if len(data) == 0:
            return np.array([])

        process_noise = float(process_noise)
        measurement_noise = float(measurement_noise)
        dt = float(dt)

        # Matrices
        A = np.array([[1, dt], [0, 1]])
        H = np.array([[1, 0]])
        Q = process_noise * np.eye(2)
        R = np.array([[measurement_noise]])
        x = np.zeros((2, 1))
        P = np.eye(2)

        results = []
        for z in data:
            # Prediction
            x_pred = A @ x
            P_pred = A @ P @ A.T + Q

            # Update
            K = P_pred @ H.T @ np.linalg.inv(H @ P_pred @ H.T + R)
            x = x_pred + K @ (z - H @ x_pred)
            P = (np.eye(2) - K @ H) @ P_pred

            results.append(x[0, 0])

        return np.array(results)

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
        """1D Extended Kalman Filter (nonlinear update)."""
        if len(data) == 0:
            return np.array([])

        process_noise = float(process_noise)
        measurement_noise = float(measurement_noise)
        dt = float(dt)

        # Nonlinear functions
        def f(x):
            return np.array([[x[0, 0] + x[1, 0]**2 * dt], [x[1, 0]]])

        def F_jacobian(x):
            return np.array([[1, 2 * x[1, 0] * dt], [0, 1]])

        H = np.array([[1, 0]])
        Q = process_noise * np.eye(2)
        R = np.array([[measurement_noise]])
        x = np.zeros((2, 1))
        P = np.eye(2)

        results = []
        for z in data:
            # Prediction
            F = F_jacobian(x)
            x_pred = f(x)
            P_pred = F @ P @ F.T + Q

            # Update
            K = P_pred @ H.T @ np.linalg.inv(H @ P_pred @ H.T + R)
            x = x_pred + K @ (z - H @ x_pred)
            P = (np.eye(2) - K @ H) @ P_pred

            results.append(x[0, 0])

        return np.array(results)

    @staticmethod
    def unscented_kalman(data, process_noise, measurement_noise, dt):
        """1D Unscented Kalman Filter (UKF) using sigma points."""
        from filterpy.kalman import UnscentedKalmanFilter as UKF
        from filterpy.kalman import MerweScaledSigmaPoints

        if len(data) == 0:
            return np.array([])

        process_noise = float(process_noise)
        measurement_noise = float(measurement_noise)
        dt = float(dt)

        def fx(x, dt):
            return np.array([x[0] + x[1]*dt, x[1]])

        def hx(x):
            return np.array([x[0]])

        points = MerweScaledSigmaPoints(2, alpha=0.1, beta=2.0, kappa=0)
        ukf = UKF(dim_x=2, dim_z=1, dt=dt, fx=fx, hx=hx, points=points)
        ukf.x = np.zeros(2)
        ukf.P *= 1
        ukf.Q = process_noise * np.eye(2)
        ukf.R = np.array([[measurement_noise]])

        results = []
        for z in data:
            ukf.predict()
            ukf.update(z)
            results.append(ukf.x[0])

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
            taper = np.hanning(len(mask) * 2)[len(mask):]
            mask *= taper
        elif window_type == "hamming":
            taper = np.hamming(len(mask) * 2)[len(mask):]
            mask *= taper

        # Apply mask
        filtered_spectrum = spectrum * mask

        # Inverse FFT
        filtered_signal = np.fft.irfft(filtered_spectrum, n=n)

        return filtered_signal

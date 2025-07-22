import numpy as np
from scipy import stats
from scipy.interpolate import interp1d


class TimeSeriesPreprocessor:
    def __init__(self, method: str = "iqr", interpolation: str = "linear"):
        """
        Initializes a preprocessing utility for 1D time-series.

        Args:
            method: Outlier detection method ('iqr' or 'zscore').
            interpolation: Interpolation method ('linear', 'cubic', etc.).
        """
        self.method = method
        self.interpolation = interpolation

    def remove_outliers(self, data_series, **kwargs):
        if self.method == "iqr":
            return self._remove_outliers_iqr(data_series, **kwargs)
        elif self.method == "zscore":
            return self._remove_outliers_zscore(data_series, **kwargs)

    def interpolate(self, data_series):
        data = np.array(data_series)
        valid_indices = np.where(~np.isnan(data))[0]
        if len(valid_indices) < 2:
            return data

        interp_func = interp1d(
            valid_indices,
            data[valid_indices],
            kind=self.interpolation,
            fill_value="extrapolate",
        )
        return interp_func(np.arange(len(data)))

    def clean(self, data_series, **kwargs):
        cleaned = self.remove_outliers(data_series, **kwargs)
        return self.interpolate(cleaned)

    @staticmethod
    def _remove_outliers_iqr(data_series, iqr_multiplier=1.5):
        data = np.array(data_series)
        q1, q3 = np.percentile(data[~np.isnan(data)], [25, 75])
        iqr = q3 - q1
        lower = q1 - iqr_multiplier * iqr
        upper = q3 + iqr_multiplier * iqr
        mask = (data < lower) | (data > upper)
        return np.where(mask, np.nan, data)

    @staticmethod
    def _remove_outliers_zscore(data_series, z_threshold=3.0):
        data = np.array(data_series)
        z_scores = np.abs(stats.zscore(data[~np.isnan(data)]))
        mask = np.zeros_like(data, dtype=bool)
        mask[~np.isnan(data)] = z_scores > z_threshold
        return np.where(mask, np.nan, data)

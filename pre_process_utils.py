# preprocessing_utils.py
import numpy as np
from scipy.interpolate import interp1d
from scipy import stats


def remove_outliers_iqr(data_series, iqr_multiplier=1.5):
    """
    Removes outliers using the Interquartile Range (IQR) method.
    Args:
        data_series: Input data (list or numpy array).
        iqr_multiplier: Threshold multiplier (default: 1.5).
    Returns:
        Cleaned data with outliers replaced by NaN.
    """
    data = np.array(data_series)
    q1, q3 = np.percentile(data[~np.isnan(data)], [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - iqr_multiplier * iqr
    upper_bound = q3 + iqr_multiplier * iqr
    outlier_mask = (data < lower_bound) | (data > upper_bound)
    data_clean = np.where(outlier_mask, np.nan, data)
    return data_clean


def remove_outliers_zscore(data_series, z_threshold=3.0):
    """
    Removes outliers using Z-score method.
    Args:
        data_series: Input data (list or numpy array).
        z_threshold: Z-score cutoff (default: 3.0).
    Returns:
        Cleaned data with outliers replaced by NaN.
    """
    data = np.array(data_series)
    z_scores = np.abs(stats.zscore(data[~np.isnan(data)]))
    outlier_mask = np.zeros_like(data, dtype=bool)
    outlier_mask[~np.isnan(data)] = z_scores > z_threshold
    data_clean = np.where(outlier_mask, np.nan, data)
    return data_clean


def interpolate_missing_values(data_series, kind="linear"):
    """
    Interpolates NaN/missing values in a time series.
    Args:
        data_series: Input data (list or numpy array).
        kind: Interpolation method ('linear', 'cubic', etc.).
    Returns:
        Interpolated data without NaNs.
    """
    data = np.array(data_series)
    valid_indices = np.where(~np.isnan(data))[0]
    if len(valid_indices) < 2:
        return data  # Not enough points to interpolate

    interp_func = interp1d(
        valid_indices,
        data[valid_indices],
        kind=kind,
        fill_value="extrapolate",
    )
    return interp_func(np.arange(len(data)))

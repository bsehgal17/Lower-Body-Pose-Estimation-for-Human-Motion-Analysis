import numpy as np
from typing import List
from post_processing.pre_process_utils import (
    remove_outliers_iqr,
    interpolate_missing_values,
)


def interpolate_series(
    data_series: List[float],
    iqr_multiplier: float = 1.5,
    interpolation_kind: str = "linear",
) -> np.ndarray:
    """
    Removes outliers using IQR method and interpolates missing values
    in a time-series signal.

    Args:
        data_series: A list of numeric values representing a 1D time-series.
        iqr_multiplier: Multiplier used to determine IQR outlier boundaries.
        interpolation_kind: Type of interpolation to apply (e.g., 'linear', 'cubic').

    Returns:
        A NumPy array of the cleaned and interpolated time-series data.
    """
    cleaned = remove_outliers_iqr(data_series, iqr_multiplier)
    interpolated = interpolate_missing_values(cleaned, kind=interpolation_kind)
    return np.array(interpolated)

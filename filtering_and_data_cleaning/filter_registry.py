from .filter_library import FilterLibrary

FILTER_FN_MAP = {
    "butterworth": FilterLibrary.butterworth,
    "gaussian": FilterLibrary.gaussian,
    "median": FilterLibrary.median,
    "moving_average": FilterLibrary.moving_average,
    "savitzky": FilterLibrary.savitzky_golay,
    "wiener": FilterLibrary.wiener,
    "kalman": FilterLibrary.kalman,
    "gvcspl": FilterLibrary.gvcspl,
    "ekf": FilterLibrary.extended_kalman,
    "ukf": FilterLibrary.unscented_kalman,
}

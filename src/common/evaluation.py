"""Small evaluation utilities shared, byte-for-byte, across the experiment scripts."""

import numpy as np


def mean_absolute_error(actual: np.ndarray, forecast: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - forecast)))


def arrange_quantiles(values: np.ndarray, expected_shape: tuple[int, ...]) -> tuple[np.ndarray, int]:
    quantiles = np.asarray(values, dtype=np.float64)
    if quantiles.shape != expected_shape:
        raise ValueError(f"Unexpected quantile shape {quantiles.shape}; expected {expected_shape}.")
    if not np.isfinite(quantiles).all():
        raise FloatingPointError("Forecast quantiles contain NaN or infinity.")
    crossing_count = int(np.count_nonzero(np.diff(quantiles, axis=-1) < 0))
    if crossing_count:
        quantiles = np.sort(quantiles, axis=-1)
    return quantiles, crossing_count

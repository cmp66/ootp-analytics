"""Quantile helpers of the outlook scores; numpy only, so that the viewer can use them without the model libraries."""

import numpy as np

QUANTILE_LEVELS = np.array([0.1, 0.25, 0.5, 0.75, 0.9])


def exceed_probability(quantiles: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Chance that a value is above a threshold, from its quantiles at 10, 25, 50, 75 and 90 % (rows): linear between them and
    along the end segments outside them. A threshold of -inf is always exceeded, +inf never."""
    q = np.sort(np.asarray(quantiles, dtype=float), axis=1)
    t = np.asarray(thresholds, dtype=float)
    levels = QUANTILE_LEVELS
    cdf = np.full(len(q), np.nan)
    width = np.maximum(np.diff(q, axis=1), 1e-9)
    for k in range(4):
        mask = (t >= q[:, k]) & (t < q[:, k + 1])
        cdf[mask] = levels[k] + (t[mask] - q[mask, k]) / width[mask, k] * (levels[k + 1] - levels[k])
    low = t < q[:, 0]
    slope = (levels[1] - levels[0]) / width[:, 0]
    cdf[low] = np.clip(levels[0] - (q[low, 0] - t[low]) * slope[low], 0.0, levels[0])
    high = t >= q[:, 4]
    slope_high = (levels[4] - levels[3]) / width[:, 3]
    cdf[high] = np.clip(levels[4] + (t[high] - q[high, 4]) * slope_high[high], levels[4], 1.0)
    cdf[np.isneginf(t)] = 0.0
    cdf[np.isposinf(t)] = 1.0
    return 1.0 - cdf

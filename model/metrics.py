"""Error measures that weight every row by the volume behind its target."""

import numpy as np


def _arrays(y, pred, w):
    y, pred, w = (np.asarray(a, dtype=float) for a in (y, pred, w))
    if not (len(y) == len(pred) == len(w)):
        raise ValueError("y, predictions and weights must have the same length")
    if len(y) == 0 or w.sum() <= 0:
        raise ValueError("no rows with weight to score")
    return y, pred, w


def weighted_mae(y, pred, w) -> float:
    y, pred, w = _arrays(y, pred, w)
    return float(np.average(np.abs(y - pred), weights=w))


def weighted_rmse(y, pred, w) -> float:
    y, pred, w = _arrays(y, pred, w)
    return float(np.sqrt(np.average((y - pred) ** 2, weights=w)))


def weighted_r2(y, pred, w) -> float:
    """1 - weighted squared error / weighted variance of y (can be negative: worse than predicting the mean)."""
    y, pred, w = _arrays(y, pred, w)
    spread = np.sum(w * (y - np.average(y, weights=w)) ** 2)
    return float(1.0 - np.sum(w * (y - pred) ** 2) / spread) if spread > 0 else float("nan")


def calibration(y, pred, w, bins: int = 10):
    """Mean prediction and mean outcome (weighted) in equal-weight bins of the prediction.

    A calibrated model has the two columns close to each other; returns rows of (mean prediction, mean outcome, weight).
    """
    y, pred, w = _arrays(y, pred, w)
    order = np.argsort(pred)
    y, pred, w = y[order], pred[order], w[order]
    edges = np.searchsorted(np.cumsum(w) / w.sum(), np.linspace(0, 1, bins + 1)[1:-1])
    rows = []
    for idx in np.split(np.arange(len(y)), edges):
        if len(idx):
            rows.append((np.average(pred[idx], weights=w[idx]), np.average(y[idx], weights=w[idx]), w[idx].sum()))
    return np.array(rows)


def summarize(y, pred, w) -> dict:
    return {"r2": weighted_r2(y, pred, w), "mae": weighted_mae(y, pred, w), "rmse": weighted_rmse(y, pred, w)}

import numpy as np
import pytest

from model import metrics


def test_weighted_errors_by_hand():
    y = np.array([1.0, 2.0, 4.0])
    pred = np.array([1.0, 3.0, 2.0])
    w = np.array([1.0, 1.0, 2.0])
    assert metrics.weighted_mae(y, pred, w) == pytest.approx((0 + 1 + 2 * 2) / 4)
    assert metrics.weighted_rmse(y, pred, w) == pytest.approx(np.sqrt((0 + 1 + 2 * 4) / 4))
    mean = (1 + 2 + 8) / 4
    spread = 1 * (1 - mean) ** 2 + 1 * (2 - mean) ** 2 + 2 * (4 - mean) ** 2
    assert metrics.weighted_r2(y, pred, w) == pytest.approx(1 - 9 / spread)


def test_weights_change_the_score_and_perfect_predictions_score_one():
    y = np.array([0.0, 10.0])
    pred = np.array([0.0, 0.0])
    assert metrics.weighted_mae(y, pred, [1, 1]) == pytest.approx(5)
    assert metrics.weighted_mae(y, pred, [9, 1]) == pytest.approx(1)
    assert metrics.weighted_r2(y, y, [3, 1]) == 1.0
    assert metrics.weighted_r2(y, np.full(2, 5.0), [1, 1]) == pytest.approx(0.0)  # the mean is worth R^2 = 0


def test_bad_inputs_are_rejected():
    with pytest.raises(ValueError):
        metrics.weighted_mae([1, 2], [1], [1, 1])
    with pytest.raises(ValueError):
        metrics.weighted_r2([], [], [])
    with pytest.raises(ValueError):
        metrics.weighted_mae([1], [1], [0])


def test_calibration_bins_hold_equal_weight_and_follow_the_prediction():
    rng = np.random.default_rng(0)
    pred = rng.normal(size=1000)
    y = pred + rng.normal(scale=0.1, size=1000)
    w = rng.integers(1, 5, 1000)
    rows = metrics.calibration(y, pred, w, bins=5)
    assert len(rows) == 5
    assert np.all(np.diff(rows[:, 0]) > 0)
    assert np.allclose(rows[:, 0], rows[:, 1], atol=0.1)
    assert rows[:, 2].sum() == w.sum()
    assert rows[:, 2].max() / rows[:, 2].min() < 1.2

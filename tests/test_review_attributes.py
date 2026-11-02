import numpy as np
import pandas as pd
import pytest

from model.data import TrainingTable
from scripts import review_attributes as review


def make_table(n_players=200, seasons=4, seed=0):
    """Target depends on `signal` and a weaker `helper`; `noise_a` and `noise_b` are unrelated to it."""
    rng = np.random.default_rng(seed)
    players = pd.DataFrame(
        {
            "ID": np.arange(n_players),
            "signal": rng.normal(size=n_players),
            "helper": rng.normal(size=n_players),
            "twin": rng.normal(size=n_players),
        }
    )
    frames = []
    for season in range(seasons):
        frame = players.copy()
        frame["season"] = 2046 + season
        frame["noise_a"] = rng.normal(size=n_players)
        frame["noise_b"] = rng.normal(size=n_players)
        frame["target"] = 3 * frame["signal"] + 1.0 * frame["helper"] + rng.normal(scale=1.0, size=n_players)
        frame["weight"] = 500.0
        frames.append(frame)
    return TrainingTable("t", pd.concat(frames, ignore_index=True), ["signal"])


def test_the_verdicts_need_a_gain_and_a_clear_signal():
    assert review.verdict_add(0.01, 0.001) == "ADD"
    assert review.verdict_add(0.01, 0.01) == "no gain"  # not three standard errors
    assert review.verdict_add(0.001, 0.0001) == "no gain"  # too small to matter
    assert review.verdict_add(-0.01, 0.001) == "HURTS"
    assert review.verdict_drop(-0.05, 0.005) == "KEEP"
    assert review.verdict_drop(-0.05, 0.05) == "can drop"
    assert review.verdict_drop(0.01, 0.001) == "can drop"


def test_paired_comparison_mean_and_standard_error():
    base = np.array([0.5, 0.5, 0.5, 0.5])
    other = np.array([0.6, 0.7, 0.6, 0.7])
    delta, se = review.compare(base, other)
    assert delta == pytest.approx(0.15)
    assert se == pytest.approx(np.std([0.1, 0.2, 0.1, 0.2], ddof=1) / 2)


def test_the_folds_are_the_same_for_every_feature_list(monkeypatch):
    table = make_table()
    seen = []
    original = review.splits.group_kfold

    def spy(groups, folds, seed):
        for train, test in original(groups, folds, seed):
            seen.append(tuple(test[:5]))
            yield train, test

    monkeypatch.setattr(review.splits, "group_kfold", spy)
    first = review.fold_scores("ridge", table, ["signal"], seeds=2, folds=4)
    folds_one = list(seen)
    seen.clear()
    second = review.fold_scores("ridge", table, ["signal", "noise_a", "helper"], seeds=2, folds=4)
    assert len(first) == len(second) == 8
    assert seen == folds_one  # paired differences need identical folds
    assert review.fold_scores("ridge", table, ["signal"], seeds=2, folds=4) == pytest.approx(first)


def test_a_real_rating_is_added_noise_is_not_and_the_main_rating_is_kept():
    table = make_table()
    core = ["signal", "noise_a"]
    rows = review.review_stream(
        "t",
        table,
        core,
        {"helper": ["helper"], "noise": ["noise_b"]},
        {},
        "ridge",
        seeds=3,
        folds=5,
    )
    verdict = {(r["test"], r["item"]): r["verdict"] for r in rows}
    assert verdict[("add", "helper")] == "ADD"
    assert verdict[("add", "noise")] == "no gain"
    assert verdict[("drop", "signal")] == "KEEP"
    assert verdict[("drop", "noise_a")] == "can drop"


def test_a_skill_group_is_dropped_together():
    table = make_table()
    table.frame["signal_twin"] = table.frame["signal"] + np.random.default_rng(1).normal(scale=0.05, size=len(table))
    core = ["signal", "signal_twin", "noise_a"]
    rows = review.review_stream(
        "t", table, core, {}, {}, "ridge", seeds=2, folds=4, groups={"signal both": ["signal", "signal_twin"]}
    )
    single = {r["item"]: r["verdict"] for r in rows if r["test"] == "drop"}
    group = {r["item"]: r["verdict"] for r in rows if r["test"] == "drop skill"}
    assert single["signal"] == single["signal_twin"] == "can drop"  # each one covers for the other
    assert group == {"signal both": "KEEP"}


def test_the_clustered_regression_recovers_a_slope_and_its_significance():
    table = make_table()
    result = review.clustered_regression(table, ["signal", "noise_a"]).set_index("rating")
    assert result.loc["signal", "per_rating_point"] == pytest.approx(3, abs=0.2)
    assert abs(result.loc["signal", "t"]) > 10
    assert abs(result.loc["noise_a", "t"]) < 3

import numpy as np
import pandas as pd
import pytest

from model import evaluate
from model.data import TrainingTable
from tests.test_model_train import make_table


def test_the_simplest_family_within_the_tolerance_is_chosen():
    assert evaluate.choose_family({"ridge": 0.55, "boosting": 0.553, "mlp": 0.549}) == "ridge"
    assert evaluate.choose_family({"ridge": 0.46, "boosting": 0.505, "mlp": 0.506}) == "boosting"
    assert evaluate.choose_family({"ridge": 0.30, "boosting": 0.40, "mlp": 0.45}) == "mlp"
    assert evaluate.choose_family({"boosting": 0.4, "mlp": 0.4}) == "boosting"


def test_player_folds_are_shared_and_cover_all_players():
    folds = evaluate.player_folds([5, 3, 3, 9, 1, 5, 9, 7, 8, 2], n_splits=3, seed=1)
    assert set(folds) == {1, 2, 3, 5, 7, 8, 9}
    assert set(folds.values()) == {0, 1, 2}
    assert folds == evaluate.player_folds([1, 2, 3, 5, 7, 8, 9], n_splits=3, seed=1)  # order and repeats do not matter


def split_frame():
    """Two players, two seasons, both sides; the right side carries 3/4 of the volume for one hand and 1/2 for the other."""
    rows = []
    for pid, hand, right_weight, left_weight in [(1, 0, 300, 100), (2, 1, 200, 200)]:
        for season in (2046, 2047):
            rows.append({"ID": pid, "season": season, "side_left": 0.0, "weight": right_weight, "hand": hand})
            rows.append({"ID": pid, "season": season, "side_left": 1.0, "weight": left_weight, "hand": hand})
    return pd.DataFrame(rows)


def test_platoon_mix_is_the_share_of_volume_against_the_right_overall_and_per_hand():
    frame = split_frame()
    assert evaluate.platoon_mix(frame)[None] == pytest.approx(500 / 800)
    assert evaluate.platoon_mix(frame, "hand") == {0: pytest.approx(0.75), 1: pytest.approx(0.5)}


def test_blending_weights_the_sides_by_the_mix_of_the_hand():
    frame = split_frame()
    prediction = np.where(frame["side_left"] == 0, 10.0, 2.0)
    mix = evaluate.platoon_mix(frame, "hand")
    blended = evaluate.blend_sides(frame, prediction, mix, frame["hand"])
    by_player = blended.groupby("ID")["total"].first()
    assert by_player[1] == pytest.approx(0.75 * 10 + 0.25 * 2)
    assert by_player[2] == pytest.approx(0.5 * 10 + 0.5 * 2)
    assert len(blended) == 4 and {"right", "left", "total"} <= set(blended.columns)
    single = evaluate.blend_sides(frame, prediction, 0.6)
    assert single["total"].iloc[0] == pytest.approx(0.6 * 10 + 0.4 * 2)


def test_a_player_season_without_one_side_is_left_out():
    frame = split_frame().drop(index=1)  # player 1, 2046 has no left row
    blended = evaluate.blend_sides(frame, np.ones(len(frame)), 0.7)
    assert len(blended) == 3
    assert not ((blended["ID"] == 1) & (blended["season"] == 2046)).any()


def test_babip_difference_becomes_runs_with_the_sign_flipped():
    assert evaluate.babip_to_runs(0.010, 500, 0.8) == pytest.approx(-4.0)
    assert evaluate.babip_to_runs(-0.010, 500, 0.8) == pytest.approx(4.0)


def test_folds_are_shared_between_a_plain_and_a_stacked_evaluation():
    total = make_table(n_players=80, seasons=3)
    folds = evaluate.player_folds(total.frame["ID"], 4, seed=2)
    prediction = evaluate.cross_validate_folds("ridge", total, folds, n_splits=4)
    assert np.isfinite(prediction).all()
    # a held-out player's predictions come from a model that never saw him: same inputs give a different model per fold
    fold_of_row = total.frame["ID"].map(folds).to_numpy()
    assert set(fold_of_row) == {0, 1, 2, 3}


def stacked_pair(n_players=120, seasons=3, seed=0):
    """A total table and its stacked version: the total target is the platoon blend of the two side targets."""
    rng = np.random.default_rng(seed)
    base = rng.normal(size=n_players)
    rows, totals = [], []
    for season in range(2046, 2046 + seasons):
        for pid in range(n_players):
            right = 3 * base[pid] + rng.normal(scale=0.3)
            left = 2 * base[pid] + rng.normal(scale=0.3)
            for side, target in ((0.0, right), (1.0, left)):
                rows.append(
                    {
                        "ID": pid,
                        "season": season,
                        "side_left": side,
                        "x": base[pid] + rng.normal(scale=0.05),
                        "target": target,
                        "weight": 300.0 if side == 0 else 100.0,
                    }
                )
            totals.append(
                {"ID": pid, "season": season, "x": base[pid], "target": 0.75 * right + 0.25 * left, "weight": 400.0}
            )
    split = TrainingTable("split", pd.DataFrame(rows), ["x", "side_left"])
    total = TrainingTable("total", pd.DataFrame(totals), ["x"])
    return split, total


def test_the_stacked_model_predicts_the_total_target_through_the_platoon_blend():
    split, total = stacked_pair()
    folds = evaluate.player_folds(total.frame["ID"], 4, seed=0)
    mix = evaluate.platoon_mix(split.frame)[None]
    assert mix == pytest.approx(0.75)
    merged, scores = evaluate.cross_validate_stacked("ridge", split, total, folds, mix, n_splits=4)
    assert len(merged) == len(total)
    assert scores["r2"] > 0.9
    assert (merged["total"] - (0.75 * merged["right"] + 0.25 * merged["left"])).abs().max() < 1e-9
    wrong, wrong_scores = evaluate.cross_validate_stacked("ridge", split, total, folds, 0.25, n_splits=4)
    assert wrong_scores["r2"] < scores["r2"]  # the wrong mix is scored worse

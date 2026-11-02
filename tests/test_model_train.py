import numpy as np
import pandas as pd
import pytest

from model import splits
from model.data import TrainingTable
from model.train import (
    BoostingModel,
    MLPConfig,
    MLPModel,
    RidgeModel,
    Scaler,
    cross_validate,
    fit_table,
    holdout_last_seasons,
    make_model,
)

FEATURES = ["a", "b", "c"]


def make_table(n_players=120, seasons=6, noise=0.3, seed=0, nonlinear=False, player_effect=0.0):
    """Ratings that are nearly constant for a player over the seasons, a target that depends on them."""
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(n_players, 3))
    rows = []
    effect = rng.normal(size=n_players) * player_effect
    for season in range(2046, 2046 + seasons):
        ratings = base + rng.normal(scale=0.05, size=base.shape)
        signal = 2 * ratings[:, 0] - 3 * ratings[:, 1]
        if nonlinear:
            signal = signal + 4 * ratings[:, 0] * ratings[:, 2] + 2 * np.sin(2 * ratings[:, 1])
        target = signal + effect + rng.normal(scale=noise, size=n_players)
        frame = pd.DataFrame(ratings, columns=FEATURES)
        frame["ID"] = np.arange(1, n_players + 1)
        frame["season"] = season
        frame["target"] = target
        frame["weight"] = rng.integers(100, 600, n_players).astype(float)
        rows.append(frame)
    return TrainingTable("synthetic", pd.concat(rows, ignore_index=True), list(FEATURES))


FAST_MLP = dict(hidden=(32, 16), max_epochs=60, patience=10, batch_size=128, device="cpu")


def fitted(family, table, **params):
    params = FAST_MLP if family == "mlp" and not params else params
    return fit_table(make_model(family, **params), table, seed=1)


def test_scaler_uses_the_training_rows_and_keeps_constant_columns_finite():
    X = np.array([[1.0, 5.0], [3.0, 5.0], [5.0, 5.0]])
    scaler = Scaler(X, w=np.array([1.0, 1.0, 2.0]))
    out = scaler.transform(X)
    assert np.average(out[:, 0], weights=[1, 1, 2]) == pytest.approx(0)
    assert np.isfinite(out).all() and (out[:, 1] == 0).all()


@pytest.mark.parametrize("family", ["ridge", "boosting", "mlp"])
def test_a_prediction_depends_on_the_row_only_not_on_the_batch(family):
    table = make_table(n_players=60, seasons=3)
    model = fitted(family, table)
    X = table.X.iloc[:40]
    together = model.predict(X)
    alone = np.concatenate([model.predict(X.iloc[[i]]) for i in range(5)])
    # the old code re-fitted the scaler on whatever was predicted, so a single row gave a different answer
    assert alone == pytest.approx(together[:5], abs=1e-4)
    shifted = X.copy() + 5.0
    assert model.predict(pd.concat([X.iloc[:5], shifted])).shape == (45,)
    assert model.predict(pd.concat([X.iloc[:5], shifted]))[:5] == pytest.approx(together[:5], abs=1e-4)


@pytest.mark.parametrize("family", ["ridge", "boosting", "mlp"])
def test_columns_are_matched_by_name_and_bad_input_is_rejected(family):
    table = make_table(n_players=50, seasons=3)
    model = fitted(family, table)
    X = table.X.iloc[:10]
    assert model.predict(X[["c", "a", "b"]]) == pytest.approx(model.predict(X), abs=1e-6)
    with pytest.raises(KeyError, match="missing"):
        model.predict(X[["a", "b"]])
    broken = X.copy()
    broken.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="missing or infinite"):
        model.predict(broken)


def test_ridge_recovers_a_known_linear_relationship():
    table = make_table(noise=0.01, seasons=3)
    model = fitted("ridge", table, alpha=1e-6)
    pred = model.predict(table.X)
    assert np.corrcoef(pred, table.y)[0, 1] > 0.999


def test_weights_pull_the_fit_toward_heavy_rows():
    frame = pd.DataFrame({"a": [0.0, 0.0, 0.0, 0.0], "b": 0.0, "c": 0.0, "ID": [1, 2, 3, 4], "season": 2046})
    frame["target"] = [10.0, 10.0, 0.0, 0.0]
    frame["weight"] = [1.0, 1.0, 100.0, 100.0]
    table = TrainingTable("w", frame, list(FEATURES))
    model = fit_table(RidgeModel(alpha=1e-9), table)
    assert model.predict(table.X)[0] == pytest.approx(20 / 202, abs=1e-6)  # weighted mean, near 0 not 5


def test_the_network_learns_a_nonlinear_target():
    table = make_table(nonlinear=True, noise=0.1)
    _, scores = cross_validate("mlp", table, n_splits=3, **dict(FAST_MLP, max_epochs=150, patience=20))
    _, ridge = cross_validate("ridge", table, n_splits=3)
    assert scores["r2"] > 0.8
    assert scores["r2"] > ridge["r2"] + 0.05


def test_the_same_seed_gives_the_same_network_and_another_seed_a_different_one():
    table = make_table(n_players=60, seasons=3)
    first = fit_table(make_model("mlp", **FAST_MLP), table, seed=3).predict(table.X.iloc[:20])
    again = fit_table(make_model("mlp", **FAST_MLP), table, seed=3).predict(table.X.iloc[:20])
    other = fit_table(make_model("mlp", **FAST_MLP), table, seed=4).predict(table.X.iloc[:20])
    assert first == pytest.approx(again, abs=1e-6)
    assert not np.allclose(first, other, atol=1e-4)


def test_stopping_rows_come_from_players_held_out_of_training():
    table = make_table(n_players=60, seasons=4)
    model = fitted("mlp", table)
    groups = table.frame["ID"].to_numpy()
    stop_players = set(groups[model.val_index])
    train_players = set(groups[model.train_index])
    assert stop_players and train_players
    assert len(model.train_index) + len(model.val_index) == len(groups)
    assert not (stop_players & train_players)  # no player is both trained on and used for stopping


def test_a_player_effect_cannot_be_memorized_across_the_split():
    # the target is a per-player constant unrelated to the ratings' meaning: only memorizing a player helps
    table = make_table(n_players=100, seasons=6, noise=0.05, player_effect=3.0)
    table.frame["target"] = table.frame.groupby("ID")["target"].transform("mean")  # the same value in every season
    _, grouped = cross_validate("boosting", table, n_splits=4)
    rows = table.frame
    random_train, random_test = _random_row_split(len(rows))
    leaked = fit_table(BoostingModel(), table, random_train, seed=0)
    leaked_r2 = (
        np.corrcoef(leaked.predict(rows.iloc[random_test][FEATURES]), rows.iloc[random_test]["target"])[0, 1] ** 2
    )
    assert leaked_r2 > 0.5  # a random row split looks good: the same player is on both sides
    assert grouped["r2"] < leaked_r2 - 0.2 and grouped["r2"] < 0.5


def _random_row_split(n, seed=0):
    order = np.random.default_rng(seed).permutation(n)
    half = n // 2
    return np.sort(order[:half]), np.sort(order[half:])


def test_cross_validation_predicts_every_row_once():
    table = make_table(n_players=60, seasons=3)
    prediction, scores = cross_validate("ridge", table, n_splits=4)
    assert np.isfinite(prediction).all() and len(prediction) == len(table)
    assert set(scores) == {"r2", "mae", "rmse"}


def test_holdout_scores_only_the_last_seasons():
    table = make_table(n_players=60, seasons=6)
    prediction, scores = holdout_last_seasons("ridge", table, n_seasons=2)
    assert len(prediction) == (table.frame["season"] >= 2050).sum()
    assert scores["r2"] > 0.5


def test_unknown_family_and_loss_are_rejected():
    with pytest.raises(ValueError, match="unknown model family"):
        make_model("forest")
    table = make_table(n_players=40, seasons=2)
    with pytest.raises(ValueError, match="unknown loss"):
        fit_table(MLPModel(MLPConfig(loss="bogus", device="cpu", max_epochs=2)), table)


def test_player_holdout_import_is_the_one_used_for_stopping():
    # the stopping split is the same function the evaluation code uses for player hold-outs
    assert splits.player_holdout.__module__ == "model.splits"

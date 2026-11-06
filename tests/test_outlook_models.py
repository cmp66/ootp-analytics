import numpy as np
import pandas as pd
import pytest

from model import artifact
from model import outlook_data as outlook
from model import outlook_models as models
from scripts import build_outlook_tables, train_models, train_outlook_models


def test_pinball_brier_logloss_and_auc_match_hand_calculations():
    y = np.array([1.0, 2.0, 3.0])
    assert models.pinball(y, np.array([2.0, 2.0, 2.0]), 0.9) == pytest.approx(
        (0.9 * 1 * 0 + 0 + 0.9 * 1) / 3 * 0 + (0.1 * 1 + 0 + 0.9 * 1) / 3
    )
    labels, probability = np.array([0.0, 0.0, 1.0, 1.0]), np.array([0.1, 0.4, 0.35, 0.8])
    assert models.auc(labels, probability) == pytest.approx(0.75)
    assert models.brier(labels, probability) == pytest.approx(np.mean([0.01, 0.16, 0.4225, 0.04]))
    assert models.log_loss(labels, np.array([0.5] * 4)) == pytest.approx(np.log(2))
    assert np.isnan(models.auc(np.zeros(3), np.array([0.1, 0.2, 0.3])))


def test_calibration_table_groups_rows_by_prediction():
    p = np.linspace(0, 1, 100)
    y = (p > 0.5).astype(float)
    table = models.calibration_table(y, p, 4)
    assert table["n"].tolist() == [25, 25, 25, 25] and table["observed"].tolist() == [0.0, 0.0, 1.0, 1.0]
    assert table["predicted"].is_monotonic_increasing


def test_split_seasons_leaves_a_gap_so_training_targets_do_not_reach_the_test_seasons():
    seasons = np.repeat(np.arange(1, 41), 3)
    train, test = models.split_seasons(seasons, 3)
    assert test.tolist() == list(range(32, 38))  # the latest seasons whose 3-season future is in the table
    assert train.max() == 21 and train.max() + models.GAP < test.min()
    train5, test5 = models.split_seasons(seasons, 5)
    assert test5.max() == 35 and train5.max() + 5 < test5.min()
    career_train, career_test = models.split_seasons(seasons, models.CAREER)  # looks CAREER_SPAN (10) seasons ahead
    assert career_test.max() == 40 - models.CAREER_SPAN and career_train.max() + models.GAP < career_test.min()


def test_subsample_keeps_whole_players():
    frame = pd.DataFrame({"ID": np.repeat(np.arange(100), 10), "x": 1.0})
    sample = models.subsample_players(frame, 300)
    counts = sample.groupby("ID").size()
    assert (counts == 10).all() and 200 <= len(sample) <= 400
    assert models.subsample_players(frame, 5000) is frame


def test_probability_calibration_pulls_low_predictions_up_to_the_observed_rate():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 0.2, 20000)
    y = (rng.uniform(0, 1, 20000) < 2 * p).astype(float)  # the model predicts half of the true chance
    calibration = models.fit_probability_calibration(y, p)
    fixed = models.apply_probability_calibration(p, calibration)
    assert fixed.mean() == pytest.approx(y.mean(), rel=0.01) and p.mean() < 0.6 * y.mean()
    assert np.all(
        np.diff(models.apply_probability_calibration(np.linspace(0, 0.2, 50), calibration)) >= 0
    )  # still ordered
    test = pd.DataFrame({"season": np.repeat(np.arange(1, 5), 5000)})
    check = models.probability_calibration_check(test, y, p)
    assert check["brier_calibrated"] < check["brier_raw"]
    assert abs(check["mean_calibrated"] - check["base_rate"]) < abs(check["mean_raw"] - check["base_rate"])


class FixedModel:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=float)

    def predict(self, X):
        return self.values[: len(X)]


def test_quantile_predictions_are_sorted_so_no_band_is_upside_down():
    fitted = {0.1: FixedModel([5, 1]), 0.5: FixedModel([3, 2]), 0.9: FixedModel([1, 3])}
    out = models.predict_quantiles(fitted, pd.DataFrame({"a": [0, 0]}))
    assert out[0.1].tolist() == [1.0, 1.0] and out[0.5].tolist() == [3.0, 2.0] and out[0.9].tolist() == [5.0, 3.0]


def test_band_shift_widens_a_too_narrow_band_to_the_requested_coverage():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, 5000)
    lower, upper = np.full(5000, -0.5), np.full(5000, 0.5)  # covers about 38 %
    assert models.coverage(y, lower, upper) < 0.45
    shift = models.band_shift(y, lower, upper, 0.8)
    assert shift > 0 and models.coverage(y, lower, upper, shift) == pytest.approx(0.8, abs=0.01)
    wide = models.band_shift(y, lower - 5, upper + 5, 0.8)
    assert wide < 0  # a band that is too wide can be narrowed


def test_probability_model_returns_the_positive_class_and_zero_when_it_never_occurs():
    class Stub:
        def __init__(self, classes):
            self.classes_ = classes

        def predict_proba(self, X):
            return np.tile(np.array([0.7, 0.3])[: len(self.classes_)], (len(X), 1))

    X = pd.DataFrame({"a": [1, 2, 3]})
    assert models.ProbabilityModel(Stub([0, 1])).predict(X).tolist() == [0.3, 0.3, 0.3]
    assert models.ProbabilityModel(Stub([0])).predict(X).tolist() == [0.0, 0.0, 0.0]


def test_age_and_slot_baseline_uses_the_training_quantiles_and_falls_back_to_the_overall_one():
    train = pd.DataFrame({"slot": [0.0] * 100, "Age": [20] * 100, "dvalue_h1": np.arange(100.0)})
    baseline = models.age_slot_quantiles(train, 1, (0.1, 0.5, 0.9))
    rows = pd.DataFrame({"slot": [0.0, 5.0], "Age": [20, 40]})
    median = models.baseline_predict(baseline, rows, 0.5)
    assert median[0] == pytest.approx(49.5) and median[1] == pytest.approx(
        49.5
    )  # unseen slot and age: the overall median


def test_features_start_with_the_slot_and_only_hitters_have_position_ratings():
    hitters, pitchers = models.feature_names("hitters"), models.feature_names("pitchers")
    assert hitters[0] == pitchers[0] == "slot" and "slot_rating" in hitters and "slot_rating" not in pitchers
    assert {"WE_ord", "INT_ord", "Prone3_ord"} <= set(
        hitters
    ) and "LEA_ord" not in hitters  # the placebo traits stay out
    assert "STU" in pitchers and "STU" not in hitters


def mini_tables():
    """Two hitters at SS over three seasons: player 1 becomes top 10 then top 50, player 2 leaves after the first season."""
    players = pd.DataFrame(
        {
            "ID": [1, 2, 1, 1], "season": [1, 1, 2, 3], "Age": [20, 20, 21, 22], "AgeFine": [20.3, 20.6, 21.3, 22.3],
            "LPOS": "SS", "level": [3.0, 3.0, 2.0, 1.0],
            "WE_ord": 1.0, "INT_ord": 1.0, "Prone3_ord": 1.0, "time_share": [0.9, 0.5, 0.8, 0.95],
            "observed_h1": [True, True, True, False], "observed_h2": [True, True, False, False],
            "observed_h3": False, "observed_h5": False, "stay_h1": [1.0, 0.0, 1.0, np.nan],
        }
    )  # fmt: skip
    values = pd.DataFrame(
        {
            "ID": [1, 2, 1, 1], "season": [1, 1, 2, 3], "slot": "SS", "value": [-10.0, -20.0, 5.0, 12.0],
            "pct": [0.9, 0.4, 0.97, 0.99], "tier": ["rest", "rest", "top50", "top10"],
        }
    )  # fmt: skip
    every = values[["ID", "season", "slot", "value"]].assign(qualified=True)
    return {"players": players, "values": values, "slot_values": every}


def test_family_table_targets_come_from_later_seasons_and_gone_players_count_as_not_ranked(monkeypatch):
    monkeypatch.setattr(
        models,
        "load_ratings",
        lambda save, seasons, columns: pd.DataFrame(
            {"ID": [1, 2], "season": [1, 1], **{c: 50.0 for c in dict.fromkeys(columns)}}
        ),
    )
    table = models.family_table("x", "hitters", (1, 2), mini_tables())
    ss = table[table["slot"] == models.SLOT_CODES["SS"]].set_index(["ID", "season"])
    assert ss.loc[(1, 1), "dvalue_h1"] == 15.0 and ss.loc[(1, 2), "dvalue_h1"] == 7.0  # value in t+1 minus value now
    assert ss.loc[(1, 1), "top50_h1"] == 1.0 and ss.loc[(1, 1), "top10_h1"] == 0.0
    assert ss.loc[(1, 1), "top10_h2"] == 1.0 and ss.loc[(1, 1), "dvalue_h2"] == 22.0
    assert np.isnan(ss.loc[(2, 1), "dvalue_h1"]) and ss.loc[(2, 1), "top50_h1"] == 0.0  # gone: no value, not top 50
    assert np.isnan(ss.loc[(1, 3), "top50_h1"])  # t+1 is beyond the table


def test_player_table_and_availability_targets_do_not_look_back():
    tables = mini_tables()
    table = models.add_player_targets(models.player_table(tables["players"], tables["values"]), (1,))
    row = table.set_index(["ID", "season"])
    assert row.loc[(1, 1), "avail_h1"] == 0.8 and row.loc[(1, 2), "avail_h1"] != row.loc[(1, 2), "time_share"]
    assert np.isnan(row.loc[(1, 3), "avail_h1"]) and row.loc[(1, 1), "is_pitcher"] == 0.0


@pytest.mark.integration
def test_the_outlook_models_train_store_and_report_on_the_fixture_seasons(tmp_path, monkeypatch, capsys):
    from tests.test_model_registry import multi_season_root

    multi_season_root(tmp_path, monkeypatch)
    monkeypatch.setattr(outlook, "OUTLOOK_CONFIG", tmp_path / "outlook.toml")
    shares = {t: {"L": 0.1, "N": 0.7, "H": 0.2} for t in outlook.formats.TRAITS}
    shares["Prone"] = {"Durable": 0.1, "Normal": 0.7, "Fragile": 0.2}
    monkeypatch.setattr(outlook, "export_trait_shares", lambda ratings: shares)
    monkeypatch.setattr(models, "GAP", 1)
    monkeypatch.setattr(models, "TEST_SEASONS", 2)
    assert train_models.main(["Model-Sim", "--start", "2046", "--stop", "2050", "--no-cv"]) == 0
    assert build_outlook_tables.main(["Model-Sim"]) == 0
    capsys.readouterr()
    report = tmp_path / "report.md"
    assert train_outlook_models.main(["Model-Sim", "--horizons", "1", "--max-rows", "5000", "--out", str(report)]) == 0
    text = report.read_text()
    for heading in ("## Value:", "## Tier chances", "## Staying in the league and availability"):
        assert heading in text
    names = sorted(p.stem for p in (artifact.models_dir("Model-Sim", "Standard27") / "outlook").glob("*.joblib"))
    assert "value-hitters-h1-q10" in names and "value-pitchers-h1-q90" in names and "top50-hitters-h1" in names
    assert "stay-player-h1" in names  # the fixture has no playing time, so no availability models
    assert not any(
        (artifact.models_dir("Model-Sim", "Standard27")).glob("outlook*.joblib")
    )  # kept out of the plan 14 folder
    loaded = artifact.load_artifact("Model-Sim", "Standard27", "outlook/top50-hitters-h1")
    table = models.family_table("Model-Sim", "hitters", (1,))
    probability = loaded.predict(table[loaded.features])
    assert probability.shape == (len(table),) and ((probability >= 0) & (probability <= 1)).all()
    assert loaded.notes["calibration"] is not None and len(loaded.notes["calibration"]["x"]) > 1
    band = artifact.load_artifact("Model-Sim", "Standard27", "outlook/value-hitters-h1-q10")
    assert "band_shift" in band.notes and band.notes["quantile"] == 0.1
    assert train_outlook_models.main(["Nope"]) == 2


def test_career_targets_are_the_peak_and_the_ever_top_flags_within_the_span(monkeypatch):
    monkeypatch.setattr(models, "CAREER_SPAN", 3)
    keyed = pd.DataFrame(
        {
            "ID": [1, 2, 3],
            "slot_name": ["SS", "SS", "SS"],
            "season": [1, 1, 1],
            "value_now": [0.0, 10.0, 5.0],
            "observed_h3": [True, True, False],
        }
    )
    every = pd.DataFrame(
        {"ID": [1, 1, 1, 3, 3], "slot": ["SS"] * 5, "season": [2, 3, 4, 2, 3], "value": [1.0, 7.0, 4.0, 9.0, 9.0]}
    )  # player 2 is gone; player 3's future is not fully observed
    values = pd.DataFrame(
        {"ID": [1, 1, 1, 3], "slot": ["SS"] * 4, "season": [2, 3, 4, 2], "tier": ["top50", "top10", "rest", "top10"]}
    )
    out = models.add_career_targets(keyed, every, values)
    assert out.loc[0, "dvalue_h99"] == 7.0 and out.loc[0, "top10_h99"] == 1.0 and out.loc[0, "top50_h99"] == 1.0
    assert np.isnan(out.loc[1, "dvalue_h99"]) and out.loc[1, "top10_h99"] == 0.0 and out.loc[1, "top50_h99"] == 0.0
    assert (
        out.loc[2, [f"{c}_h99" for c in ("dvalue", "top10", "top50")]].isna().all() and not out.loc[2, "observed_h99"]
    )
    assert models.reach(models.CAREER) == 3 and models.reach(5) == 5


def test_the_trained_horizons_are_the_ones_in_use_plus_the_career_view():
    assert models.TRAINED_HORIZONS == (1, 5, 7, 10)
    assert models.OUTLOOK_HORIZONS == (1, 5, 7, 10, models.CAREER)

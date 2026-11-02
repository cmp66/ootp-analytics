import numpy as np
import pandas as pd
import pytest

from model import outlook_data as outlook
from scripts import build_outlook_tables, review_outlook as review, train_models
from tests.test_model_registry import multi_season_root


def brute_force_cluster_se(y, x, groups):
    xtx_inv = np.linalg.inv(x.T @ x)
    beta = xtx_inv @ x.T @ y
    resid = y - x @ beta
    meat = np.zeros((x.shape[1], x.shape[1]))
    for g in np.unique(groups):
        rows = groups == g
        score = x[rows].T @ resid[rows]
        meat += np.outer(score, score)
    n, k, clusters = x.shape[0], x.shape[1], len(np.unique(groups))
    correction = clusters / (clusters - 1) * (n - 1) / (n - k)
    return beta, np.sqrt(np.diag(correction * xtx_inv @ meat @ xtx_inv))


def test_cluster_ols_matches_a_brute_force_calculation_and_gives_the_r2():
    rng = np.random.default_rng(1)
    groups = np.repeat(np.arange(60), 7)
    x = np.column_stack([np.ones(420), rng.normal(size=420), rng.normal(size=420)])
    y = x @ np.array([1.0, 2.0, -0.5]) + rng.normal(size=60)[groups] + rng.normal(size=420)
    beta, se, r2 = review.cluster_ols(y, x, groups)
    expected_beta, expected_se = brute_force_cluster_se(y, x, groups)
    assert beta == pytest.approx(expected_beta) and se == pytest.approx(expected_se)
    residual = y - x @ beta
    assert r2 == pytest.approx(1 - (residual**2).sum() / ((y - y.mean()) ** 2).sum())


def synthetic_pairs(n=6000, seed=0):
    """Pairs of seasons for hitters aged 16-22 where work ethic and missed time change the yearly change of 'BABIP'."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "ID": np.arange(n),
            "Age": rng.integers(16, 23, n),
            "LPOS": "SS",
            "C": 0.0,
            "BABIP": rng.choice(np.arange(20, 66, 5), n).astype(float),
            "BA P": 0.0,
            "WE_cat": rng.choice(["L", "N", "H"], n, p=[0.2, 0.6, 0.2]),
            "INT_cat": rng.choice(["L", "N", "H"], n),
            "LEA_cat": rng.choice(["L", "N", "H"], n),
            "LOY_cat": rng.choice(["L", "N", "H"], n),
            "FIN_cat": rng.choice(["L", "N", "H"], n),
            "Prone3_cat": rng.choice(["Durable", "Normal", "Fragile"], n),
            "time_share_next": rng.uniform(0, 1, n),
        }
    )
    frame["WE"] = frame["WE_cat"].map({"L": 30.0, "N": 100.0, "H": 170.0})
    frame["INT"] = frame["INT_cat"].map({"L": 30.0, "N": 100.0, "H": 170.0})
    frame["BA P"] = frame["BABIP"] + rng.choice([0.0, 10.0, 20.0], n)
    change = (
        1.0
        - 0.3 * (frame["WE_cat"] == "L")
        + 0.1 * (frame["WE_cat"] == "H")
        + 0.8 * frame["time_share_next"]
        + 0.02 * (frame["BA P"] - frame["BABIP"])
        + rng.normal(0, 1.0, n)
    )
    frame["BABIP_next"] = frame["BABIP"] + change
    frame["time_share_next"] = frame["time_share_next"]
    return frame


def test_development_effects_recover_injected_effects_and_find_no_placebo_effect():
    rows = review.development_effects(synthetic_pairs(), "BABIP", "BA P", "hitters")
    assert [r["group"] for r in rows] == ["young (16-22)"]  # no prime or older players in the sample
    row = rows[0]
    assert row["WE_L"] == pytest.approx(-0.3, abs=0.1) and row["WE_L_t"] < -4
    assert row["WE_H"] == pytest.approx(0.1, abs=0.1)
    assert row["time_next"] == pytest.approx(0.8, abs=0.12) and row["time_next_t"] > 10
    assert row["gap"] == pytest.approx(0.2, abs=0.08)  # 0.02 per point is 0.2 per 10 points
    for placebo in (
        "INT_L",
        "INT_H",
        "LEA_L",
        "LEA_H",
        "LOY_L",
        "LOY_H",
        "FIN_L",
        "FIN_H",
        "Prone3_Durable",
        "Prone3_Fragile",
    ):
        assert abs(row[f"{placebo}_t"]) < 3.5, placebo
    assert (
        abs(row["r2_tiers"] - row["r2_numbers"]) < 0.02
    )  # tiers keep what the numbers carry (the effect is tier-shaped)


def test_applies_to_picks_hitters_catchers_pitchers_and_pitches_that_exist_in_both_seasons():
    frame = pd.DataFrame(
        {
            "LPOS": ["SS", "C", "SP", "RP"],
            "C": [0.0, 45.0, 0.0, 0.0],
            "FB": [0.0, 0.0, 50.0, 0.0],
            "FB_next": [0.0, 0.0, 55.0, 50.0],
        }
    )
    assert review.applies_to(frame, "hitters", "x").tolist() == [True, True, False, False]
    assert review.applies_to(frame, "catchers", "x").tolist() == [False, True, False, False]
    assert review.applies_to(frame, "pitchers", "x").tolist() == [False, False, True, True]
    assert review.applies_to(frame, "pitchers_with", "FB").tolist() == [False, False, True, False]


def test_season_pairs_join_only_consecutive_seasons_of_the_same_player():
    panel = pd.DataFrame(
        {
            "ID": [1, 1, 1, 2, 2],
            "season": [10, 11, 13, 10, 12],
            "Age": [20, 21, 23, 30, 32],
            "BABIP": [40, 45, 55, 50, 50],
        }
    )
    pairs = review.season_pairs(panel)
    assert pairs[["ID", "season", "BABIP", "BABIP_next", "Age_next"]].values.tolist() == [[1, 10, 40, 45, 21]]


def test_value_aging_measures_the_drop_against_the_spread_of_mlb_players():
    rng = np.random.default_rng(0)
    ids = np.arange(400)
    now = pd.DataFrame({"ID": ids, "season": 1, "slot": "DH", "value": rng.normal(0, 10, 400)})
    later = pd.DataFrame({"ID": ids, "season": 2, "slot": "DH", "value": now["value"] - 8.0})  # everybody loses 8
    players = pd.DataFrame({"ID": np.tile(ids, 2), "season": np.repeat([1, 2], 400), "Age": 28, "level": 1})
    aging = review.value_aging(players, pd.concat([now, later]))
    row = aging[aging["slot"] == "DH"].iloc[0]
    assert row["change"] == pytest.approx(-8.0) and row["p_drop"] == 1.0  # 8 is more than half the spread of about 14


@pytest.mark.integration
def test_the_review_runs_end_to_end_on_the_fixture_seasons(tmp_path, monkeypatch, capsys):
    multi_season_root(tmp_path, monkeypatch)
    monkeypatch.setattr(outlook, "OUTLOOK_CONFIG", tmp_path / "outlook.toml")
    shares = {t: {"L": 0.1, "N": 0.7, "H": 0.2} for t in outlook.formats.TRAITS}
    shares["Prone"] = {"Durable": 0.1, "Normal": 0.7, "Fragile": 0.2}
    monkeypatch.setattr(outlook, "export_trait_shares", lambda ratings: shares)
    assert train_models.main(["Model-Sim", "--start", "2046", "--stop", "2050", "--no-cv"]) == 0
    assert build_outlook_tables.main(["Model-Sim"]) == 0
    capsys.readouterr()
    out = tmp_path / "report.md"
    assert review.main(["Model-Sim", "--out", str(out), "--csv-dir", str(tmp_path / "csv")]) == 0
    text = out.read_text()
    for heading in (
        "## 1 How attributes develop",
        "## 2 Staying in the league",
        "## 3 Aging of value and the baseline",
    ):
        assert heading in text
    assert (tmp_path / "csv" / "Model-Sim-baseline.csv").exists()
    assert review.main(["Nope"]) == 2

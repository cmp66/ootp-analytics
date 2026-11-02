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


def test_cluster_ols_without_enough_rows_returns_nan_instead_of_warning():
    beta, se, r2 = review.cluster_ols(np.array([1.0]), np.ones((1, 2)), np.array([1]))
    assert np.isnan(beta).all() and np.isnan(se).all() and np.isnan(r2)


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


def entry_fixture():
    """Three entrants and later seasons: A is top 50 three seasons on, B is gone, C is ranked but not top 50."""
    players = pd.DataFrame(
        {
            "ID": [1, 2, 3, 1, 3, 1, 3],
            "season": [10, 10, 10, 13, 13, 15, 15],
            "Age": [18, 18, 22, 21, 25, 23, 27],
            "left_truncated": False,
            "entry_season": [10, 10, 10, 10, 10, 10, 10],
            "entry_route": [
                "hs_draft",
                "hs_draft",
                "college_draft",
                "hs_draft",
                "college_draft",
                "hs_draft",
                "college_draft",
            ],
            "WE_cat": "N",
            "INT_cat": "N",
            "Prone3_cat": "Normal",
            "observed_h1": True,
            "observed_h3": [True, True, True, True, True, False, False],
            "observed_h5": [True, True, True, False, False, False, False],
        }
    )
    values = pd.DataFrame(
        {
            "ID": [1, 2, 3, 1, 3, 1, 3],
            "season": [10, 10, 10, 13, 13, 15, 15],
            "slot": "SS",
            "value": [-20.0, -30.0, -5.0, 8.0, -4.0, 10.0, -3.0],
            "tier": ["rest", "rest", "rest", "top50", "rest", "top10", "rest"],
        }
    )
    return players, values


def test_best_by_player_season_and_entrant_outcomes_do_not_look_back():
    players, values = entry_fixture()
    best = review.best_by_player_season(values)
    assert best.set_index(["season", "ID"]).loc[(13, 1), ["best_value", "top50", "top10"]].tolist() == [
        8.0,
        True,
        False,
    ]
    new = review.entrants(players, best, horizons=(3, 5))
    new = new[new["season"] == 10].set_index("ID")
    assert new.loc[1, "top50_h3"] == 1.0 and new.loc[3, "top50_h3"] == 0.0
    assert new.loc[2, "top50_h3"] == 0.0  # gone by season 13: counts as not top 50
    assert new.loc[1, "top10_h5"] == 1.0  # season 15
    assert new.loc[3, "top50_h5"] == 0.0 and new.loc[1, "entry_value"] == -20.0


def test_top50_at_age_counts_missing_players_as_not_top_50_and_skips_unobserved_ages():
    players, values = entry_fixture()
    best = review.best_by_player_season(values)
    new = review.entrants(players, best, horizons=(3, 5))
    new = new[new["season"] == 10].assign(entry_pct=[30.0, 30.0, 80.0])
    assert review.top50_at_age(players, best, new, 21).empty  # three entrants: too few for any row of the report
    full = review.top50_at_age(players, best, new.loc[new.index.repeat(60)], 21)
    hs = full[full["route"] == "hs_draft"].iloc[0]
    assert hs["present"] == pytest.approx(0.5) and hs["top50"] == pytest.approx(0.5)  # A is there and top 50, B is gone


def test_route_tables_need_enough_entrants_and_age_adds_to_the_value():
    rng = np.random.default_rng(0)
    n = 3000
    age = rng.integers(17, 23, n)
    entry = rng.normal(-30, 15, n)
    frame = pd.DataFrame(
        {
            "ID": np.arange(n),
            "Age": age,
            "entry_route": np.where(age < 20, "hs_draft", "college_draft"),
            "entry_value": entry,
            "entry_pct": np.clip(50 + entry, 0, 100),
            "top50_h3": (entry + rng.normal(0, 10, n) > -10).astype(float),
            "top50_h5": (entry + rng.normal(0, 10, n) > -10).astype(float),
            "top10_h5": 0.0,
            "value_h3": entry + 3.0 * (age - 20) + rng.normal(0, 5, n),
            "WE_cat": "N",
            "INT_cat": "N",
            "Prone3_cat": "Normal",
        }
    )
    by_value = review.route_by_value(frame)
    low = by_value[by_value["band"] == "< -30"]["top50_h5"].mean()
    high = by_value[by_value["band"].isin(["0 to 10", "10 to 20", "20+"])]["top50_h5"].mean()
    assert low < 0.2 < 0.8 < high or pd.isna(high)
    matters = review.age_matters(frame).set_index("model")["r2"]
    assert matters["+ age"] > matters["entry value only"] + 0.05  # age was built into the future value


def test_missed_time_is_lower_for_fragile_players_at_equal_value():
    rng = np.random.default_rng(2)
    n = 6000
    tier = rng.choice(["Durable", "Normal", "Fragile"], n)
    players = pd.DataFrame(
        {
            "ID": np.arange(n),
            "season": 1,
            "Age": rng.integers(22, 35, n),
            "primary_level": 1.0,
            "Prone3_cat": tier,
            "Prone": np.where(tier == "Fragile", 140.0, np.where(tier == "Durable", 20.0, 80.0)),
            "time_share": np.clip(
                0.8 + 0.05 * (tier == "Durable") - 0.1 * (tier == "Fragile") + rng.normal(0, 0.1, n), 0, 1
            ),
        }
    )
    best = pd.DataFrame({"ID": np.arange(n), "season": 1, "best_value": rng.normal(0, 5, n)})
    table = review.missed_time(players, best).set_index(["group", "tier"])
    assert (
        table.loc[("26-30", "Fragile"), "relative_to_normal"]
        < 0.95
        < 1.0
        < table.loc[("26-30", "Durable"), "relative_to_normal"]
    )
    effects = review.missed_time_effects(players, best)
    assert effects.loc["Fragile", "effect"] == pytest.approx(-0.1, abs=0.02) and effects.loc[
        "Durable", "effect"
    ] == pytest.approx(0.05, abs=0.02)

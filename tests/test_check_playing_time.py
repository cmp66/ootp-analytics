import numpy as np
import pandas as pd

from scripts import check_playing_time as cpt


def synthetic_panel(
    effect: float, selection: float, players: int = 1500, seasons: int = 6, seed: int = 0
) -> pd.DataFrame:
    """Hitters whose six batting ratings change by a growth trend of their own, noise, plus `effect` points per 100 PA played that
    season; the manager gives the
    next season's PA to the players who improved (`selection` x the change, in 100 PA)."""
    rng = np.random.default_rng(seed)
    rows = []
    for player in range(players):
        rating, pa_next, trend = rng.normal(45, 6), rng.uniform(100, 500), rng.normal(0, 0.8)
        for season in range(seasons):
            pa = pa_next
            change = trend + effect * pa / 100 + rng.normal(0, 1.0)
            rating += change if season else 0
            # the PA of the next season follow the change just made
            pa_next = float(np.clip(0.6 * pa + 120 + 100 * selection * change + rng.normal(0, 60), 0, 700))
            row = {
                "ID": player,
                "season": 2000 + season,
                "LPOS": "SS",
                "AgeFine": 22.0,
                "WE": 100,
                "INT": 100,
                "pa": pa,
            }
            row.update({c: rating for c in cpt.HIT} | {c: rating + 5 for c in cpt.HIT_POT})
            row["level"] = 1.0
            rows.append(row)
    return pd.DataFrame(rows)


def test_clustered_ols_recovers_the_coefficients():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(4000, 2))
    y = 1.5 * x[:, 0] - 0.5 * x[:, 1] + rng.normal(0, 0.5, 4000)
    beta, se, rows, r2 = cpt.clustered_ols(y, np.column_stack([x, np.ones(4000)]), np.arange(4000) // 4)
    assert rows == 4000 and np.allclose(beta, [1.5, -0.5, 0], atol=0.05) and (se < 0.05).all() and r2 > 0.9


def test_the_table_pairs_consecutive_seasons_and_leaves_gaps_out():
    panel = synthetic_panel(0.0, 0.0, players=30)
    panel = panel[~((panel["ID"] == 0) & (panel["season"] == 2002))]  # player 0 misses a season
    table = cpt.development_table(panel, "hitters")
    assert not ((table["ID"] == 0) & (table["season"] == 2003)).any()  # the change over a gap is not used
    first = table[(table["ID"] == 1) & (table["season"] == 2002)].iloc[0]
    assert first["prev_pt"] == panel[(panel["ID"] == 1) & (panel["season"] == 2001)]["pa"].iloc[0]
    assert first["next_pt"] == panel[(panel["ID"] == 1) & (panel["season"] == 2003)]["pa"].iloc[0]


def test_selection_is_seen_in_the_placebo_and_pushes_the_raw_estimate_up_and_the_full_one_down():
    table = cpt.development_table(synthetic_panel(effect=0.0, selection=0.8), "hitters")
    raw, full = cpt.estimate(table, terms=("pt",)), cpt.estimate(table)
    assert (
        raw["pt"][0] > 0.1
    )  # PA are persistent and the change goes on from year to year: the raw estimate is too high
    assert full["next_pt"][0] > 0.2  # the PA of the next season carry the selection (the placebo)
    assert full["pt"][0] < raw["pt"][0]  # and holding them fixed gives the lower bound
    assert cpt.verdict(raw["pt"], full["next_pt"]).endswith("placebo shows selection")


def test_a_real_effect_without_selection_has_a_flat_placebo_and_a_raw_estimate_near_the_effect():
    table = cpt.development_table(synthetic_panel(effect=0.3, selection=0.0), "hitters")
    raw, full = cpt.estimate(table, terms=("pt",)), cpt.estimate(table)
    assert 0.25 < raw["pt"][0] < 0.35 and abs(full["next_pt"][0]) < 0.05
    assert "placebo is flat" in cpt.verdict(raw["pt"], full["next_pt"])


def test_the_verdict_names_the_upper_bound_per_season_and_works_without_a_placebo():
    assert cpt.verdict((0.1, 0.005), None) == "upper bound +0.60 points per 600, no placebo"
    assert "placebo shows selection" in cpt.verdict((0.1, 0.005), (0.2, 0.005))
    assert "placebo is flat" in cpt.verdict((0.1, 0.005), (0.001, 0.005))

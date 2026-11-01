import numpy as np
import pandas as pd
import pytest

from stats import pitching
from tests.helpers import (
    batting_league_totals,
    league_table,
    make_batting,
    make_pitching,
    make_ratings,
    pitching_league_totals,
)


def run_pitching(df, df_lg_woba=None):
    lg = league_table(pitching_league_totals(df))
    return (
        pitching.calculate_player_pitching_stats(
            df, lg, make_ratings(df["ID"]), "Test", df_lg_woba
        ),
        lg,
    )


def test_role_classification():
    df = pd.DataFrame({"G": [10, 10, 10, 1], "GS": [10, 0, 4, 0]})
    assert pitching.classify_pitcher_role(df).tolist() == ["SP", "RP", "Mix", "RP"]


def test_waa_sums_to_zero_within_each_role():
    out, _ = run_pitching(make_pitching())
    assert set(out["PRole"]) == {"SP", "RP", "Mix"}
    for role, group in out.groupby("PRole"):
        assert group["WAA"].sum() == pytest.approx(0, abs=1e-9), role


def test_league_totals_survive_a_missing_role():
    df = pd.DataFrame({"PRole": ["RP", "RP"], "IPClean": [20.0, 10.0], "R": [8, 3]})
    totals = pitching.calculate_league_totals(df)
    assert np.isnan(totals.loc["SP", "RA9"])
    assert totals.loc["RP", "RA9"] == pytest.approx(11 / 30 * 9)
    assert totals.loc["Total", "RA9"] == pytest.approx(11 / 30 * 9)


def test_rate_columns():
    df = make_pitching()
    out, _ = run_pitching(df)
    assert out["KPercent"].to_numpy() == pytest.approx(
        (df["K"] / df["BF"] * 100).to_numpy()
    )
    assert out["BBPercent"].to_numpy() == pytest.approx(
        (df["BB"] / df["BF"] * 100).to_numpy()
    )
    assert out["HRPercent"].to_numpy() == pytest.approx(
        (df["HR"] / df["BF"] * 100).to_numpy()
    )


def test_innings_use_baseball_notation():
    df = make_pitching()
    out, _ = run_pitching(df)
    whole = np.floor(df["IP"].to_numpy() + 1e-9)
    thirds = np.rint((df["IP"].to_numpy() - whole) * 10)
    assert out["IPClean"].to_numpy() == pytest.approx(whole + thirds / 3)


def test_woba_against_uses_the_batting_coefficients():
    df = make_pitching()
    batting_lg = league_table(batting_league_totals(make_batting(seed=9)))
    out_default, _ = run_pitching(df)
    out_batting, _ = run_pitching(df, batting_lg)
    assert not np.allclose(out_default["wOBAa"], out_batting["wOBAa"])
    c = batting_lg["Value"]
    den = df["AB"] + df["BB"] - df["IBB"] + df["SF"] + df["HP"]
    num = (
        df["SINGLE"] * c["coef_1B"]
        + df["DOUBLE"] * c["coef_2B"]
        + df["TRIPLE"] * c["coef_3B"]
        + df["HR"] * c["coef_HR"]
        + (df["BB"] - df["IBB"]) * c["coef_BB"]
        + df["HP"] * c["coef_HP"]
    )
    assert out_batting["wOBAa"].to_numpy() == pytest.approx((num / den).to_numpy())


def test_waaadj_is_alias_of_waa():
    out, _ = run_pitching(make_pitching())
    assert out["WAAAdj"].tolist() == out["WAA"].tolist()

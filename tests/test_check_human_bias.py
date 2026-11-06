import numpy as np
import pandas as pd
import pytest

from scripts import check_human_bias as bias


def test_a_rating_with_vs_right_and_vs_left_versions_counts_once():
    groups = bias.composite(["STU vR", "STU vL", "STM", "CON.1 vR", "CON.1 vL"])
    assert groups == {"STU": ["STU vR", "STU vL"], "STM": ["STM"], "CON.1": ["CON.1 vR", "CON.1 vL"]}


def test_the_regression_recovers_the_slope_per_standard_deviation_and_its_error():
    rng = np.random.default_rng(0)
    n = 4000
    x = pd.Series(rng.normal(size=n))
    weight = rng.uniform(50, 150, n)
    y = 2.5 * x + rng.normal(scale=10, size=n)
    beta, se = bias.regress(y, np.column_stack([np.ones(n), bias.standardize(x, weight)]), weight)
    assert beta[1] == pytest.approx(2.5, abs=0.6) and 0 < se[1] < 0.5 and abs(beta[0]) < 0.6
    assert bias.standardize(pd.Series([1.0, 1.0]), [1, 1]).tolist() == [0.0, 0.0]  # no spread: no division by zero


def test_leagues_are_pooled_by_precision_and_a_rating_is_flagged_only_when_they_agree_in_sign():
    pooled, z = bias.pool([2.0, 4.0], [1.0, 1.0])
    assert pooled == pytest.approx(3.0) and z == pytest.approx(3.0 * np.sqrt(2))
    precise, _ = bias.pool([1.0, 5.0], [0.1, 10.0])
    assert precise == pytest.approx(1.0, abs=0.01)  # the precise league counts for more

    def league(coef, se):
        return {"n": 100, "sd": 5.0, "coef": {"CON.1": coef}, "se": {"CON.1": se}}

    agree = bias.table({"A": league(2.0, 0.5), "B": league(3.0, 0.5)}, {"Ref": league(0.1, 0.3)}, flag_z=2.5)
    assert agree.loc[0, "FLAG"] == "<<<" and agree.loc[0, "A"].startswith("+2.00 (+4.0)") and "ref Ref" in agree
    disagree = bias.table({"A": league(2.0, 0.5), "B": league(-3.0, 0.5)}, {}, flag_z=2.5)
    assert disagree.loc[0, "FLAG"] == ""
    weak = bias.table({"A": league(0.5, 0.5), "B": league(0.6, 0.5)}, {}, flag_z=2.5)
    assert weak.loc[0, "FLAG"] == ""  # the same sign, but not significant


def test_the_script_reports_a_league_that_is_not_configured(capsys):
    assert bias.main(["--leagues", "Nowhere"]) == 2
    assert "not in config/update.toml" in capsys.readouterr().err

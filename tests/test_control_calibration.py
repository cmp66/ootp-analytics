import numpy as np
import pandas as pd
import pytest

from model import domains, relative
from util.config import load_saves


def league():
    """Four pitchers with control 40, 50, 50 and 60 (the mean of vs-right and vs-left) and their innings."""
    ids = pd.Index([1, 2, 3, 4], name="ID")
    players = pd.DataFrame({"CON.1 vR": [40, 50, 55, 60], "CON.1 vL": [40, 50, 45, 60]}, index=ids)
    weights = {domains.PITCHING_FIP: pd.Series([100.0, 100.0, 100.0, 100.0], index=ids)}
    scored = pd.DataFrame(
        {
            "pitching_fip": 10.0, "pitching_fip_vs_right": 12.0, "pitching_fip_vs_left": 8.0, "pitching_fip_plain": 9.0,
            "pitching_babip_runs": -1.0, "pitching_runs": 9.0,
        },
        index=ids,
    )  # fmt: skip
    return scored, players, weights


def test_control_is_measured_in_standard_deviations_of_the_pitchers_who_pitched():
    _, players, weights = league()
    z = relative.control_z(players, weights[domains.PITCHING_FIP])
    sd = np.sqrt(np.mean((np.array([40, 50, 50, 60]) - 50.0) ** 2))
    assert z.round(4).tolist() == pytest.approx([-10 / sd, 0, 0, 10 / sd], abs=1e-4)
    heavy = relative.control_z(players, pd.Series([1000.0, 1.0, 1.0, 1.0], index=players.index))  # weighted by innings
    assert heavy.iloc[0] == pytest.approx(0.0, abs=0.3) and heavy.iloc[3] > 5


def test_the_calibration_adds_runs_per_standard_deviation_to_the_fip_values_and_the_total():
    scored, players, weights = league()
    out = relative.calibrate_pitching(scored, players, weights, 2.0)
    z = relative.control_z(players, weights[domains.PITCHING_FIP])
    assert out["pitching_control_adjustment"].tolist() == pytest.approx((2.0 * z).tolist())
    for column in (
        "pitching_fip",
        "pitching_fip_vs_right",
        "pitching_fip_vs_left",
        "pitching_fip_plain",
        "pitching_runs",
    ):
        assert (out[column] - scored[column]).tolist() == pytest.approx((2.0 * z).tolist())
    assert out["pitching_babip_runs"].equals(scored["pitching_babip_runs"])  # the BABIP part is not touched
    assert (
        out.loc[1, "pitching_runs"] < scored.loc[1, "pitching_runs"] < out.loc[4, "pitching_runs"]
    )  # poor control loses
    extreme = players.copy()
    extreme.loc[1, ["CON.1 vR", "CON.1 vL"]] = [
        -200,
        -200,
    ]  # a rating far outside the league (a minor leaguer, who did not pitch)
    unused = {domains.PITCHING_FIP: weights[domains.PITCHING_FIP].where(players.index != 1, 0.0)}
    capped = relative.calibrate_pitching(scored, extreme, unused, 2.0)
    assert capped.loc[1, "pitching_control_adjustment"] == pytest.approx(-2.0 * relative.MAX_CONTROL_Z)
    again = relative.calibrate_pitching(scored, players, weights, 0.0)  # a league without the setting is unchanged
    assert again.equals(scored)


def test_a_pitcher_without_a_control_rating_keeps_his_value_and_the_league_mean_stays_put():
    scored, players, weights = league()
    players.loc[2, ["CON.1 vR", "CON.1 vL"]] = np.nan
    out = relative.calibrate_pitching(scored, players, weights, 2.0)
    assert (
        out.loc[2, "pitching_fip"] == scored.loc[2, "pitching_fip"] and out.loc[2, "pitching_control_adjustment"] == 0.0
    )
    _, shifts = relative.add_relative(out, {**weights, domains.PITCHING_BABIP: weights[domains.PITCHING_FIP]})
    assert (
        abs(shifts["pitching_fip"] - 10.0) < 0.6
    )  # innings weighted z has mean zero: the league reference barely moves


def test_the_setting_is_read_per_save_and_only_the_human_leagues_have_it():
    saves = load_saves()
    assert (
        saves["Model-Sim"].pitching_control_per_sd == 0.0 and saves["Model-Sim-26-Std"].pitching_control_per_sd == 0.0
    )
    assert all(saves[name].pitching_control_per_sd == 2.0 for name in ("TGS27", "Dugout26", "PBL27"))

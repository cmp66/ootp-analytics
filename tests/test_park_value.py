import numpy as np
import pandas as pd
import pytest

from model import artifact, park
from scripts import predict_park, train_models
from tests.test_model_registry import multi_season_root
from util import paths
from viewer import parks

COEF = {"1B": 0.9, "2B": 1.25, "3B": 1.6, "HR": 2.0, "scale": 1.2}


def factors(**park_row):
    base = {
        "AVG_L": 1.0,
        "AVG_R": 1.0,
        "HR_L": 1.0,
        "HR_R": 1.0,
        "D": 1.0,
        "T": 1.0,
        "RUNS": 1.0,
        "AVG": 1.0,
        "HR": 1.0,
    }
    return pd.DataFrame({"PARK": {**base, **park_row}, "FLAT": base}).T.rename_axis("team")


def context(table, home_share=0.5):
    rates = pd.DataFrame(
        {"left_share": [0.0, 1.0, 0.0], "HR": [0.03, 0.03, 0.0], "DOUBLE": [0.04, 0.04, 0.0], "TRIPLE": [0.004, 0.004, 0.0],
         "H": [0.25, 0.25, 0.0]}, index=pd.Index([1, 2, 3], name="ID"),
    )  # fmt: skip
    return parks.ParkContext(table, home_share, COEF, 4.5, rates)


def outlook(slots=("SS", "SS", "SP"), orgs=("PARK", "PARK", "PARK")):
    return pd.DataFrame(
        {
            "ID": [1, 2, 3], "slot": list(slots), "org": list(orgs), "value_now": [10.0, 10.0, 5.0],
            "bat_value": [4.0, 4.0, np.nan],
            "pit_value": [np.nan, np.nan, 5.0], "field_value": [6.0, 6.0, np.nan], "ceiling_h99": [20.0, 20.0, 15.0],
            "p_top10_h99": [0.3, 0.3, 0.2], "season_value_h5": [9.0, 9.0, 4.0],
        }
    )  # fmt: skip


def test_a_hitter_gains_what_the_park_adds_to_his_events_with_the_run_weights_and_half_the_games():
    ctx = context(
        factors(HR_L=1.2, HR_R=1.2)
    )  # home runs +20 % in the park, +10 % over a season of which half is at home
    adjusted = parks.adjust(outlook(), ctx, team="PARK")
    # 0.03 HR per PA more by 10 %: 0.003; the hits stay the same so the singles lose those 0.003
    expected = (COEF["HR"] * 0.003 - COEF["1B"] * 0.003) * 600 / COEF["scale"]
    assert adjusted.loc[0, "park_effect"] == pytest.approx(expected) and expected > 0
    assert adjusted.loc[0, "value_now"] == pytest.approx(10 + expected) and adjusted.loc[
        0, "bat_value"
    ] == pytest.approx(4 + expected)
    assert adjusted.loc[0, "field_value"] == 6.0  # fielding is not moved
    assert adjusted.loc[0, "ceiling_h99"] == pytest.approx(20 + expected) and adjusted.loc[
        0, "season_value_h5"
    ] == pytest.approx(9 + expected)
    assert adjusted.loc[0, "p_top10_h99"] == 0.3  # a chance is not a value


def test_the_side_a_batter_hits_from_picks_the_left_or_right_factor_and_a_player_without_rates_does_not_move():
    ctx = context(factors(HR_L=1.4, HR_R=1.0))
    adjusted = parks.adjust(outlook(slots=("SS", "SS", "SS")), ctx, team="PARK")
    right, left, none = adjusted["park_effect"]
    assert left > 0 and right == pytest.approx(0)  # the lefty uses the left factor, the righty the right one
    assert none == 0  # no rates for him
    assert parks.adjust(outlook(), ctx, team="FLAT")["park_effect"].abs().max() == 0  # a neutral park changes nothing


def test_a_pitcher_allows_runs_in_proportion_to_the_run_factor_of_the_park():
    ctx = context(factors(RUNS=1.10))  # effective factor 1.05 with half the games at home
    pitcher = parks.adjust(outlook(), ctx, team="PARK").loc[2]
    base = 4.5 * 200 / 9  # runs per 200 IP of the league
    assert pitcher["value_now"] == pytest.approx(1.05 * 5.0 - 0.05 * base) and pitcher["pit_value"] == pytest.approx(
        1.05 * 5.0 - 0.05 * base
    )
    assert pitcher["park_effect"] < 0  # a hitters' park costs a pitcher runs


def test_every_team_can_be_valued_in_its_own_park_and_a_free_agent_is_left_alone():
    ctx = context(factors(HR_L=1.2, HR_R=1.2))
    frame = outlook(slots=("SS", "SS", "SS"), orgs=("PARK", "FLAT", "FA"))
    adjusted = parks.adjust(frame, ctx, own=True)
    assert (
        adjusted.loc[0, "park_effect"] > 0
        and adjusted.loc[1, "park_effect"] == 0
        and adjusted.loc[2, "park_effect"] == 0
    )
    assert frame["value_now"].tolist() == [10.0, 10.0, 5.0]  # the frame itself is not changed
    assert context(factors(HR_L=1.2, HR_R=1.2), home_share=1.0).home_share == 1.0


def test_a_switch_hitter_takes_the_share_of_the_league_for_the_left_side():
    share = park.left_share(pd.Series([1, 2, 3]), 0.72)
    assert share.tolist() == [0.0, 1.0, 0.72]


@pytest.fixture
def root(tmp_path, monkeypatch):
    return multi_season_root(tmp_path, monkeypatch)


@pytest.mark.integration
def test_the_park_models_train_store_next_to_the_value_models_and_score_every_player_with_ratings(root, capsys):
    assert train_models.main(["Model-Sim", "--start", "2046", "--stop", "2050", "--streams", "park"]) == 0
    assert "park event rates: 4 models" in capsys.readouterr().out
    folder = artifact.models_dir("Model-Sim", "Standard27")
    assert sorted(p.stem for p in (folder / "park").glob("*.joblib")) == ["DOUBLE", "H", "HR", "TRIPLE"]
    assert not list(folder.glob("*.joblib"))  # not mixed into the value models
    models = park.load_models("Model-Sim", "Standard27")
    assert models["HR"].notes["event"] == "HR" and models["HR"].metrics["mean_rate"] > 0
    assert predict_park.main(["Model-Sim", "--season", "2051"]) == 0
    out = pd.read_csv(paths.output_file("Model-Sim", 2051, "park-rates"))
    assert (
        {"ID", "left_share", "HR", "DOUBLE", "TRIPLE", "H"} <= set(out.columns)
        and out["ID"].is_unique
        and len(out) > 100
    )
    assert out[["HR", "DOUBLE", "TRIPLE", "H"]].ge(0).all().all() and out["HR"].mean() < out["H"].mean()
    with pytest.raises(FileNotFoundError):
        park.load_models("Model-Sim", "Standard99")


def test_the_value_against_one_side_uses_the_batting_side_a_switch_hitter_takes_against_it():
    rates = pd.DataFrame(
        {
            "left_share": [0.0, 1.0, 0.72],
            "HR": [0.03] * 3,
            "DOUBLE": [0.04] * 3,
            "TRIPLE": [0.004] * 3,
            "H": [0.25] * 3,
        },
        index=pd.Index([1, 2, 3], name="ID"),
    )
    ctx = parks.ParkContext(factors(HR_L=1.4, HR_R=1.0), 1.0, COEF, 4.5, rates)  # only left-handed power is helped
    frame = pd.DataFrame(
        {
            "ID": [1, 2, 3], "slot": ["SS"] * 3, "org": ["PARK"] * 3, "B": [1, 2, 3], "value_now": [10.0] * 3,
            "value_vs_right": [10.0] * 3, "value_vs_left": [10.0] * 3,
        }
    )  # fmt: skip
    out = parks.adjust(frame, ctx, team="PARK")
    gain = out["value_vs_right"] - 10
    assert (
        gain[0] == pytest.approx(0.0) and gain[1] > 0
    )  # the right-handed hitter is not helped, the left-handed one is
    assert gain[2] == pytest.approx(gain[1])  # a switch hitter bats left against a right-hander
    assert out.loc[2, "value_vs_left"] == pytest.approx(10.0)  # and right against a left-hander
    assert out.loc[1, "value_vs_left"] == pytest.approx(
        out.loc[1, "value_vs_right"]
    )  # a left-handed hitter bats left always

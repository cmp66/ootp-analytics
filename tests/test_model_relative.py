import numpy as np
import pandas as pd
import pytest

from model import relative
from scripts import predict_values, train_models
from tests.test_model_registry import multi_season_root


def ids(values):
    return pd.Series(values, index=pd.Index(range(1, len(values) + 1), name="ID"), dtype=float)


def test_weighted_mean_uses_playing_time_and_ignores_players_without_both():
    values = pd.Series({1: 10.0, 2: 0.0, 3: 99.0, 4: np.nan})
    weights = pd.Series({1: 100.0, 2: 300.0, 4: 50.0, 5: 20.0})  # 3 never played, 4 has no value, 5 is not scored
    assert relative.weighted_mean(values, weights) == pytest.approx(2.5)
    assert np.isnan(relative.weighted_mean(values, pd.Series(dtype=float)))


def test_family_of_columns():
    assert relative.family_of("batting_vs_left") == "batting"
    assert relative.family_of("batting_plain") == "batting"
    assert relative.family_of("baserunning") == "baserunning"
    assert relative.family_of("pitching_fip_vs_right") == "pitching-fip"
    assert relative.family_of("pitching_babip_runs") == "pitching-babip"
    assert relative.family_of("pitching_babip") is None  # BABIP units, only used through the runs
    assert relative.family_of("fielding_6") == "fielding_6"
    assert relative.family_of("Age") is None


def test_relative_columns_are_the_raw_ones_shifted_by_the_weighted_league_mean():
    scored = pd.DataFrame(
        {
            "batting": ids([10, 0, 30]),
            "fielding_6": ids([20, 40, np.nan]),
            "pitching_fip": ids([5, 5, 5]),
            "pitching_babip_runs": ids([1, 3, 2]),
            "pitching_runs": ids([6, 8, 7]),
            "pitching_babip": ids([0.01, 0.02, 0.03]),
        }
    )
    weights = {
        "batting": ids([100, 100, 200]),  # mean (10 + 0 + 60) / 4 = 17.5
        "fielding_6": ids([300, 100, 100]),  # player 3 has no value: (6000 + 4000) / 400 = 25
        "pitching-fip": ids([50, 50, 50]),
        "pitching-babip": ids([100, 100, 200]),  # (1 + 3 + 4) / 4 = 2
    }
    out, shifts = relative.add_relative(scored, weights)
    assert shifts["batting"] == pytest.approx(17.5)
    assert shifts["fielding_6"] == pytest.approx(25.0)
    assert shifts["pitching_runs"] == pytest.approx(5.0 + 2.0)
    assert out["batting_rel"].tolist() == pytest.approx([-7.5, -17.5, 12.5])
    assert out["fielding_6_rel"].iloc[:2].tolist() == pytest.approx([-5.0, 15.0])
    assert np.isnan(out["fielding_6_rel"].iloc[2])
    assert out["pitching_runs_rel"].tolist() == pytest.approx([-1.0, 1.0, 0.0])
    assert "pitching_babip_rel" not in out
    assert out[list(scored.columns)].equals(scored)  # the raw columns are untouched


@pytest.mark.integration
def test_predict_values_writes_rel_columns_next_to_the_raw_ones(tmp_path, monkeypatch, capsys):
    multi_season_root(tmp_path, monkeypatch)
    assert (
        train_models.main(
            ["Model-Sim", "--start", "2046", "--stop", "2050", "--no-cv", "--streams", "batting", "baserunning"]
        )
        == 0
    )
    capsys.readouterr()
    out = tmp_path / "values.csv"
    assert predict_values.main(["Model-Sim", "--season", "2051", "--out", str(out)]) == 0
    assert "league reference" in capsys.readouterr().out
    result = pd.read_csv(out)
    for column in ["batting", "batting_vs_right", "batting_plain", "baserunning"]:
        assert f"{column}_rel" in result
        shift = (result[column] - result[f"{column}_rel"]).dropna()
        assert np.ptp(shift) < 1e-9  # one league shift per column
    weights = relative.season_weights("Model-Sim", 2051)["batting"]
    played = result.set_index("ID")[["batting", "batting_rel"]].join(weights.rename("w"), how="inner").dropna()
    assert np.average(played["batting_rel"], weights=played["w"]) == pytest.approx(0, abs=1e-9)

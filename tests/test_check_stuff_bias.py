import numpy as np
import pandas as pd
import pytest

from scripts import check_stuff_bias, train_models
from tests.test_model_registry import multi_season_root


def ratings_frame(stuff, control, homers, pbabip):
    n = len(stuff)
    return pd.DataFrame(
        {
            "STU vR": stuff, "STU vL": stuff, "CON.1 vR": control, "CON.1 vL": control,
            "HRR vR": homers, "HRR vL": homers, "PBABIP vR": pbabip, "PBABIP vL": pbabip,
            "target": np.zeros(n), "weight": np.ones(n), "start_share": np.zeros(n),
        }
    )  # fmt: skip


def test_the_groups_are_stuff_only_control_only_and_balanced_by_the_mean_of_the_two_sides():
    frame = ratings_frame([70, 40, 50, 70], [40, 70, 50, 70], [40, 70, 50, 70], [40, 40, 50, 70])
    masks = check_stuff_bias.group_masks(check_stuff_bias.profile(frame), check_stuff_bias.Groups())
    assert list(masks["stuff only"]) == [True, False, False, False]  # 70 stuff, everything else at most 50
    assert list(masks["control only"]) == [False, True, False, False]  # control and home runs 70, stuff 40
    assert list(masks["balanced"]) == [False, False, True, True]  # the four within 10 points
    assert masks["all"].all()


def test_the_comparison_is_the_weighted_difference_with_its_standard_error():
    c = check_stuff_bias.compare([2.0, 4.0], [1.0, 1.0], [1.0, 3.0])  # errors 1 and 3, weights 1 and 3
    assert (
        c["diff"] == pytest.approx(2.5) and c["prediction"] == pytest.approx(1.0) and c["outcome"] == pytest.approx(3.5)
    )
    assert c["se"] > 0 and c["z"] == pytest.approx(c["diff"] / c["se"])
    empty = check_stuff_bias.compare([], [], [])
    assert empty["n"] == 0 and np.isnan(empty["diff"])


def test_a_group_the_model_is_too_kind_to_shows_a_negative_difference_in_runs_saved_and_few_rows_are_marked():
    n = 40
    frame = ratings_frame([70] * n + [50] * n, [40] * n + [50] * n, [40] * n + [50] * n, [40] * n + [50] * n)
    frame["target"] = [-3.0] * n + [0.0] * n  # the stuff-only pitchers save 3 runs less than predicted (runs saved)
    frame.loc[0, "start_share"] = 1.0
    prediction = np.zeros(2 * n)
    ratings = check_stuff_bias.profile(frame)
    lines = check_stuff_bias.report(
        frame, prediction, ratings, check_stuff_bias.roles_of(frame), check_stuff_bias.Groups(), "runs"
    )
    stuff = [line for line in lines if line.strip().startswith("stuff only") and "both" in line][0]
    assert "-3.000" in stuff and "(few rows)" not in stuff
    starters = [line for line in lines if line.strip().startswith("stuff only") and "starters" in line][0]
    assert "(few rows)" in starters  # one starter only
    assert (
        "negative actual-pred" in lines[-1] and "too kind" in lines[-1]
    )  # runs saved: below the prediction is too kind
    hits = check_stuff_bias.report(
        frame, prediction, ratings, check_stuff_bias.roles_of(frame), check_stuff_bias.Groups(), "BABIP"
    )
    assert "positive actual-pred" in hits[-1]  # hits allowed: above the prediction is too kind


@pytest.mark.integration
def test_the_script_checks_the_pitching_models_of_a_trained_league(tmp_path, monkeypatch, capsys):
    multi_season_root(tmp_path, monkeypatch)
    assert (
        train_models.main(
            ["Model-Sim", "--start", "2046", "--stop", "2050", "--no-cv", "--streams", "pitching-fip", "pitching-babip"]
        )
        == 0
    )
    capsys.readouterr()
    assert check_stuff_bias.main(["Model-Sim", "--start", "2051", "--stop", "2051"]) == 0
    text = capsys.readouterr().out
    assert "pitching-fip:" in text and "pitching-babip:" in text and "stuff only" in text and "all" in text
    assert check_stuff_bias.main(["Nope"]) == 2

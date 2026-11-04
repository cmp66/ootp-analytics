import numpy as np
import pandas as pd
import pytest

from model import scales


def test_a_column_is_on_the_scale_its_values_fit_ignoring_zeros_and_missing():
    assert scales.column_scale(pd.Series([20, 35, 80, 0, None])) == scales.S20_80
    assert scales.column_scale(pd.Series([1, 4, 10, 0])) == scales.S1_10
    assert scales.column_scale(pd.Series([2.5, 7.5, 9])) == scales.S1_10
    assert scales.column_scale(pd.Series([0, 0, None])) == scales.OTHER
    assert scales.column_scale(pd.Series([15, 40, 90])) == scales.OTHER


def test_a_league_that_mixes_the_scales_shows_which_columns_are_which():
    frame = pd.DataFrame(
        {"BA vR": [20, 25, 50, 75], "STU vR": [1, 5, 9, 9], "PIT": [3, 4, 5, 5], "other": [1, 2, 3, 3]}
    )
    found = scales.detect_scales(frame)
    assert found == {"BA vR": scales.S20_80, "STU vR": scales.S1_10}  # PIT is a count, other is not a model rating
    table = scales.profile(frame).set_index("column")
    assert table.loc["BA vR", "step"] == 5 and table.loc["STU vR", "max"] == 9


def test_the_model_ratings_exclude_the_counts_and_the_height():
    columns = scales.rating_columns()
    assert "BA vR" in columns and "IF RNG" in columns
    assert "PIT" not in columns and "HT" not in columns


# ---------------------------------------------------------------- conversion between the scales


def paired_dumps(tmp_path):
    """Two tiny dumps of one league: 25 is 1 for 3 players and 2 for 1, 30 is always 2, 0 (no rating) stays 0."""
    first, second = tmp_path / "20-80", tmp_path / "1-10"
    first.mkdir(), second.mkdir()
    ids = [1, 2, 3, 4, 5, 6]
    batting = {"player_id": ids, "team_id": 0, "league_id": 0, "position": 1, "role": 1, "bats": 1}
    for folder, power in ((first, [25, 25, 25, 25, 30, 0]), (second, [1, 1, 1, 2, 2, 0])):
        pd.DataFrame({**batting, "batting_ratings_vsr_power": power, "batting_ratings_misc_gb_hitter_type": 1}).to_csv(
            folder / "players_batting.csv", index=False
        )
        pd.DataFrame({"player_id": ids, "pitching_ratings_overall_stuff": [20] * 6}).to_csv(
            folder / "players_pitching.csv", index=False
        )
        pd.DataFrame({"player_id": ids, "fielding_ratings_infield_range": [20] * 6}).to_csv(
            folder / "players_fielding.csv", index=False
        )
    return first, second


def test_paired_counts_keep_the_ratings_that_differ_between_the_dumps_with_the_models_names(tmp_path):
    counts = scales.paired_counts(*paired_dumps(tmp_path))
    assert set(counts["column"]) == {"POW vR"}  # the type code and the identical columns are not ratings
    pairs = {(r.v20, r.v10): r.count for r in counts.itertuples()}
    assert pairs == {(0, 0): 1, (25, 1): 3, (25, 2): 1, (30, 2): 1}


def test_a_dump_pair_that_does_not_hold_the_same_players_is_refused(tmp_path):
    first, second = paired_dumps(tmp_path)
    frame = pd.read_csv(second / "players_batting.csv")
    frame.loc[0, "player_id"] = 99
    frame.to_csv(second / "players_batting.csv", index=False)
    with pytest.raises(ValueError):
        scales.paired_counts(first, second)


def test_conversion_draws_1_10_values_with_the_observed_shares_and_keeps_zero_and_missing(tmp_path):
    conversion = scales.Conversion(scales.paired_counts(*paired_dumps(tmp_path)))
    values = pd.Series([25] * 4000 + [30, 0, np.nan])
    coarse = conversion.to_1_10(values, "POW vR", np.random.default_rng(1))
    assert set(coarse[:4000]) == {1.0, 2.0} and abs((coarse[:4000] == 1).mean() - 0.75) < 0.03
    assert coarse[4000] == 2.0 and coarse[4001] == 0.0 and np.isnan(coarse[4002])
    again = conversion.to_1_10(values, "POW vR", np.random.default_rng(1))
    assert coarse.equals(again)  # the same seed gives the same ratings


def test_expected_20_80_is_the_average_value_of_each_1_10_rating(tmp_path):
    conversion = scales.Conversion(scales.paired_counts(*paired_dumps(tmp_path)))
    expected = conversion.expected_20_80(pd.Series([1, 2, 0, np.nan]), "POW vR")
    assert expected[0] == 25.0 and expected[1] == 27.5  # 1 is only 25; 2 is 25 once and 30 once
    assert expected[2] == 0.0 and np.isnan(expected[3])


def test_the_frame_helpers_touch_only_the_columns_asked_for_and_the_ratings_view_applies_to_the_tables(tmp_path):
    from model import data

    conversion = scales.Conversion(scales.paired_counts(*paired_dumps(tmp_path)))
    frame = pd.DataFrame({"POW vR": [25, 30], "other": [25, 30]})
    coarse = conversion.coarsen_frame(frame, ["POW vR"], seed=3)
    assert coarse["other"].tolist() == [25, 30] and coarse["POW vR"].max() <= 2
    assert conversion.expect_frame(coarse, ["POW vR"])["POW vR"].max() <= 30
    assert data.RATINGS_VIEW is None
    with data.ratings_view(lambda ratings, season: ratings * 2):
        assert data.RATINGS_VIEW is not None
    assert data.RATINGS_VIEW is None


def test_the_game_version_comes_from_the_ratings_set():
    assert scales.version_of("Standard27") == "OOTP27" and scales.version_of("Standard26") == "OOTP26"
    with pytest.raises(ValueError):
        scales.version_of("Standard")


def test_the_shipped_tables_cover_the_columns_the_models_read():
    for version in ("OOTP26", "OOTP27"):
        conversion = scales.Conversion.load(version)
        assert set(scales.rating_columns()) <= set(conversion.columns)


def test_a_model_set_is_named_by_ratings_and_profile_and_the_league_names_its_profile():
    from util.config import load_saves

    assert scales.model_set("Standard27") == "Standard27"
    assert scales.model_set("Standard27", "PBL27") == "Standard27@PBL27"
    saves = load_saves()
    assert saves["TGS27"].scales is None and saves["PBL27"].scales == "PBL27" and saves["Dugout26"].scales == "Dugout26"


def test_training_on_a_mix_of_scales_needs_a_league_that_has_1_10_ratings():
    from scripts import train_models

    assert train_models.main(["Model-Sim", "--scales", "Model-Sim", "--streams", "baserunning", "--no-cv"]) == 2

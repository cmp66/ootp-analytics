import pandas as pd

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

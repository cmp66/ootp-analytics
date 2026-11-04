import numpy as np
import pandas as pd
import pytest

from model import formats
from model import outlook_data as outlook
from stats import pipeline
from tests.test_model_registry import multi_season_root
from util import paths


def test_export_values_become_categories_including_the_numbers_some_exports_use():
    assert [formats.trait_label(v) for v in ["L", "n", "H", "1", "2", "3", "U"]] == ["L", "N", "H", "L", "N", "H", "N"]
    assert np.isnan(formats.trait_label("-")) and np.isnan(formats.prone_label("???"))
    assert [formats.prone_label(v) for v in ["Iron Man", "durable", "Normal", "Fragile", "WRECKED"]] == [
        "Iron Man",
        "Durable",
        "Normal",
        "Fragile",
        "Wrecked",
    ]
    assert formats.PRONE3["Iron Man"] == "Durable" and formats.PRONE3["Wrecked"] == "Fragile"


def test_label_shares_ignore_unknown_values():
    shares = formats.label_shares(pd.Series(["L", "N", "N", "H", "-"]), formats.TRAIT_LABELS)
    assert shares == {"L": 0.25, "N": 0.5, "H": 0.25}
    with pytest.raises(ValueError):
        formats.label_shares(pd.Series(["-"]), formats.TRAIT_LABELS)


def test_bounds_reproduce_the_shares_and_order_the_labels():
    values = pd.Series(np.arange(1, 201, dtype=float))
    shares = {"L": 0.1, "N": 0.6, "H": 0.3}
    bounds = formats.fit_bounds(values, shares, formats.TRAIT_LABELS)
    assert bounds == [20.0, 140.0]
    categories = formats.categorize(values, bounds, formats.TRAIT_LABELS)
    assert categories.value_counts(normalize=True).round(3).to_dict() == {"N": 0.6, "H": 0.3, "L": 0.1}
    assert categories.iloc[19] == "L" and categories.iloc[20] == "N"  # a value equal to a bound is the lower label
    assert formats.ordinal(categories, formats.TRAIT_LABELS).iloc[[0, 50, 199]].tolist() == [0.0, 1.0, 2.0]


def test_bounds_deal_with_values_piled_up_at_the_limits():
    # 5 % of the players sit exactly at 200 (the highest label wants 3 %): the bound is the closer of 199 and 200
    values = pd.Series([200.0] * 50 + list(np.linspace(1, 199, 950)))
    bounds = formats.fit_bounds(values, {"A": 0.5, "B": 0.47, "C": 0.03}, ("A", "B", "C"))
    categories = formats.categorize(values, bounds, ("A", "B", "C"))
    assert (categories == "C").sum() == 50  # the 5 % at 200 are closer to 3 % than none
    empty_first = formats.fit_bounds(pd.Series([1.0] * 40 + [50.0] * 960), {"A": 0.01, "B": 0.99}, ("A", "B"))
    assert empty_first == [0.0]  # 4 % at the floor are further from 1 % than an empty first label


def test_nan_stays_nan_and_category_means_translate_back_to_numbers():
    values = pd.Series([10.0, 100.0, 190.0, np.nan])
    categories = formats.categorize(values, [50.0, 150.0], formats.TRAIT_LABELS)
    assert categories.isna().tolist() == [False, False, False, True]
    assert formats.category_means(values, categories) == {"L": 10.0, "N": 100.0, "H": 190.0}


def test_levels_and_draft_route():
    assert formats.level_group(1, "dump") == "MLB" and formats.level_group(5, "dump") == "A"
    assert formats.level_group(9, "dump") == "other" and formats.level_group(np.nan, "dump") == "other"
    assert formats.level_group(1, "dump", free_agent=True) == "FA"
    assert [formats.level_group(v, "export") for v in ["MLB", "A+", "A-", "R+", "R", "FA", "WL", "INT"]] == [
        "MLB",
        "A",
        "A",
        "Rookie",
        "Rookie",
        "FA",
        "other",
        "other",
    ]
    assert formats.draft_route(18) == "hs_draft" and formats.draft_route(20) == "college_draft"


def test_trait_categories_from_dump_numbers_and_from_export_categories():
    bounds = {t: [50.0, 150.0] for t in formats.TRAITS} | {formats.PRONE: [30.0, 90.0]}
    numbers = pd.DataFrame(
        {**{t: [10, 100, 190] for t in formats.TRAITS}, formats.PRONE: [10, 60, 150]},
        index=pd.Index([1, 2, 3], name="ID"),
    )
    out = outlook.with_trait_categories(numbers, bounds)
    assert out["WE_cat"].tolist() == ["L", "N", "H"] and out["INT_ord"].tolist() == [0.0, 1.0, 2.0]
    assert out["Prone3_cat"].tolist() == ["Durable", "Normal", "Fragile"] and out["Prone3_ord"].tolist() == [
        0.0,
        1.0,
        2.0,
    ]
    exported = numbers.assign(
        **{f"{t} cat": ["L", "N", "H"] for t in formats.TRAITS}, **{"Prone cat": ["Iron Man", "Normal", "Wrecked"]}
    )
    again = outlook.with_trait_categories(exported, {})  # the export's own categories need no cut points
    assert again["WE_cat"].tolist() == ["L", "N", "H"] and again["Prone3_cat"].tolist() == [
        "Durable",
        "Normal",
        "Fragile",
    ]
    assert not any(c.endswith(" cat") for c in again.columns)


def test_trait_bounds_share_the_config_file_with_the_gates(tmp_path, monkeypatch):
    monkeypatch.setattr(outlook, "OUTLOOK_CONFIG", tmp_path / "outlook.toml")
    gates = {"SS": {"IF RNG": 50, "IF ARM": 40, "pot": 60}}
    outlook.write_gates("Standard27", gates, "gates note")
    assert outlook.load_trait_bounds("Standard27") is None
    outlook.write_trait_bounds("Standard27", {"WE": [33.0, 132.0], "Prone": [30.0, 87.0]}, "traits note")
    assert outlook.load_trait_bounds("Standard27") == {"WE": [33.0, 132.0], "Prone": [30.0, 87.0]}
    assert outlook.load_gates("Standard27") == gates  # still there
    outlook.write_gates("Standard27", {"SS": {"IF RNG": 45, "IF ARM": 35, "pot": 55}}, "newer gates")
    assert outlook.load_trait_bounds("Standard27") == {
        "WE": [33.0, 132.0],
        "Prone": [30.0, 87.0],
    }  # and the other way round


@pytest.mark.integration
def test_the_export_import_keeps_the_categories_next_to_the_old_codes(tmp_path, monkeypatch):
    multi_season_root(tmp_path, monkeypatch)
    pipeline.process_season("TGS27", 2046)
    frame = pd.read_csv(paths.output_file("TGS27", 2046, "player-data"), low_memory=False)
    for trait in [*formats.TRAITS, formats.PRONE, "AD"]:
        assert f"{trait} cat" in frame.columns
    assert set(frame["WE cat"].dropna()) <= {"L", "N", "H"}
    seen = frame.groupby("WE cat")["WE"].agg(lambda s: set(s)).to_dict()
    assert seen == {k: v for k, v in {"L": {40}, "N": {100}, "H": {160}}.items() if k in seen} and "N" in seen
    codes = frame.groupby("Prone cat")["Prone"].first().to_dict()
    assert codes.get("Normal") == 100 and codes.get("Iron Man", 40) == 40 and codes.get("Wrecked", 160) == 160
    shares = outlook.export_trait_shares("Standard27")
    assert set(shares) == {*formats.TRAITS, formats.PRONE} and abs(sum(shares["WE"].values()) - 1) < 1e-9
    assert set(shares[formats.PRONE]) == set(formats.PRONE3_LABELS)

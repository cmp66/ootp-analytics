import numpy as np
import pandas as pd
import pytest

from model import formats
from model import outlook_data as outlook
from model import outlook_models as models
from model import outlook_scoring as scoring
from scripts import build_outlook_tables, predict_outlook, train_models, train_outlook_models


def test_quantile_function_interpolates_between_the_known_quantiles_and_extends_the_ends():
    known = np.array([[0.0, 1.0, 2.0, 3.0, 4.0]])  # quantiles at 10, 25, 50, 75, 90 of a line
    levels = (0.1, 0.25, 0.5, 0.75, 0.9)
    out = scoring.quantile_function(np.array([0.1, 0.5, 0.9, 0.175, 0.05, 0.95]), levels, known)[0]
    assert out[:3].tolist() == [0.0, 2.0, 4.0]
    assert out[3] == pytest.approx(0.5)  # halfway between the 10th and the 25th
    assert out[4] < 0.0 and out[5] > 4.0  # beyond the ends it continues along the last segment


def test_season_value_scales_a_positive_value_by_availability_and_leaves_a_negative_one():
    value_q = np.array([[8.0, 9.0, 10.0, 11.0, 12.0], [-12.0, -11.0, -10.0, -9.0, -8.0]])
    avail = np.array([[0.4, 0.5, 0.6], [0.4, 0.5, 0.6]])  # a fragile player: he plays 0.5 where a normal one plays 0.75
    normal = np.array([0.75, 0.75])
    result = scoring.season_value_distribution(value_q, avail, normal)
    assert result["median"][0] < 10.0 * 0.8 and result["median"][0] > 10.0 * 0.5  # about 10 x 2/3
    assert result["floor"][0] < result["median"][0] < result["ceiling"][0]
    assert result["median"][1] == pytest.approx(-10.0, abs=0.3)  # below the league average: missing time costs nothing
    durable = scoring.season_value_distribution(value_q, avail + 0.25, normal)
    assert durable["median"][0] > result["median"][0] and durable["ceiling"][0] <= 12.0 * scoring.RATIO_CAP + 1e-9


def test_normalize_to_league_rescales_each_slot_to_its_expected_count_and_keeps_the_order():
    p = np.array([0.5, 0.3, 0.1, 0.05, 0.4, 0.2])
    slots = pd.Series(["SS", "SS", "SS", "SS", "C", "C"])
    out = scoring.normalize_to_league(p, slots, {"SS": 2.0, "C": 0.3}, "top10", None, None)
    assert out[:4].sum() == pytest.approx(2.0) and out[4:].sum() == pytest.approx(0.3)
    assert np.all(np.diff(out[:4]) < 0) and out[4] > out[5]
    capped = scoring.normalize_to_league(
        np.array([0.9, 0.5, 0.01]), pd.Series(["SS"] * 3), {"SS": 2.0}, "top10", None, None
    )
    assert capped.max() <= 1.0 and capped.sum() == pytest.approx(
        2.0, abs=0.05
    )  # what the cap removes goes to the others
    bigger = scoring.normalize_to_league(
        p[:4], slots[:4], {"SS": 2.0}, "top50", 60, 30
    )  # twice the teams: twice the top players
    assert bigger.sum() == pytest.approx(4.0)


def test_export_levels_map_to_the_numbers_of_the_dump_per_version():
    assert (
        formats.export_level_number("MLB", "Standard27") == 1 and formats.export_level_number("R+", "Standard27") == 6
    )
    assert formats.export_level_number("A-", "Standard26") == 5 and formats.export_level_number("A-", "Standard27") == 4
    assert np.isnan(formats.export_level_number("FA", "Standard26")) and np.isnan(
        formats.export_level_number("WL", "Standard27")
    )


@pytest.fixture
def scored(tmp_path, monkeypatch, capsys):
    from tests.test_model_registry import multi_season_root

    multi_season_root(tmp_path, monkeypatch)
    monkeypatch.setattr(outlook, "OUTLOOK_CONFIG", tmp_path / "outlook.toml")
    shares = {t: {"L": 0.1, "N": 0.7, "H": 0.2} for t in formats.TRAITS}
    shares["Prone"] = {"Durable": 0.1, "Normal": 0.7, "Fragile": 0.2}
    monkeypatch.setattr(outlook, "export_trait_shares", lambda ratings: shares)
    monkeypatch.setattr(models, "GAP", 1)
    monkeypatch.setattr(models, "TEST_SEASONS", 2)
    assert train_models.main(["Model-Sim", "--start", "2046", "--stop", "2050", "--no-cv"]) == 0
    assert build_outlook_tables.main(["Model-Sim"]) == 0
    assert (
        train_outlook_models.main(
            ["Model-Sim", "--horizons", "1", "3", "--max-rows", "5000", "--out", str(tmp_path / "r.md")]
        )
        == 0
    )
    capsys.readouterr()
    return tmp_path


@pytest.mark.integration
def test_the_scoring_path_builds_the_same_features_as_the_training_table(scored):
    inputs = scoring.season_inputs("Model-Sim", 2050)
    inputs["save"] = "Model-Sim"
    ptable = scoring.scoring_player_table(inputs["state"], inputs["ranked"])
    rows = inputs["ranked"][inputs["ranked"]["slot"].isin(models.FAMILIES["hitters"])]
    rows = rows[["ID", "season", "slot", "value", "pct", "rank", "tier"]]
    rows = rows.rename(columns={"value": "value_now", "pct": "pct_now", "rank": "rank_now", "tier": "tier_now"})
    ratings = models.load_ratings("Model-Sim", [2050], models.rating_columns_needed("hitters"))
    mine = models.family_features(rows, ptable[models.CONTEXT_COLUMNS], ratings, "hitters")
    trained = models.family_table("Model-Sim", "hitters", (1,))
    trained = trained[trained["season"] == 2050]
    columns = [c for c in models.feature_names("hitters") if c != "slot"]
    a, b = mine.set_index(["ID", "slot"])[columns].sort_index(), trained.set_index(["ID", "slot"])[columns].sort_index()
    assert a.index.equals(b.index)
    assert (
        np.abs(a.fillna(-999) - b.fillna(-999)).to_numpy().max() < 1e-3
    )  # the saved tables are rounded to four decimals


@pytest.mark.integration
def test_predict_outlook_writes_ordered_bands_and_valid_chances(scored, capsys):
    assert predict_outlook.main(["Model-Sim", "--season", "2050"]) == 0
    assert "ranked slots" in capsys.readouterr().out
    from util import paths

    slots = pd.read_csv(paths.output_file("Model-Sim", 2050, "outlook-slots"))
    players = pd.read_csv(paths.output_file("Model-Sim", 2050, "outlook-players"))
    assert not slots.duplicated(["ID", "slot"]).any() and players["ID"].is_unique
    for h in (1, 3):
        assert (slots[f"floor_h{h}"] <= slots[f"value_h{h}"] + 1e-6).all() and (
            slots[f"value_h{h}"] <= slots[f"ceiling_h{h}"] + 1e-6
        ).all()
        assert (slots[f"season_floor_h{h}"] <= slots[f"season_ceiling_h{h}"] + 1e-6).all()
        assert slots[[f"p_top10_h{h}", f"p_top50_h{h}"]].apply(lambda c: c.between(0, 1)).all().all()
        assert (slots[f"p_top10_h{h}"] <= slots[f"p_top50_h{h}"] + 1e-6).mean() > 0.95  # top 10 is a subset of top 50
        assert players[f"p_stay_h{h}"].between(0, 1).all()
    assert "depth_player" in players and set(players["depth_player"]) <= {True, False}
    assert predict_outlook.main(["Model-Sim", "--season", "2050", "--models-from", "Nope"]) == 2


def test_patch_helper_and_unknown_models_are_clear_errors(tmp_path, monkeypatch):
    from util import paths

    monkeypatch.setattr(paths, "FILES_ROOT", tmp_path)
    with pytest.raises(FileNotFoundError):
        scoring.OutlookModels("Model-Sim", "Standard27")

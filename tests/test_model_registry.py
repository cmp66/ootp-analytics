import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from model import artifact, data, domains, registry
from stats import pipeline
from util import paths

SEASONS = [2046, 2047, 2048, 2049, 2050, 2051]


def multi_season_root(tmp_path, monkeypatch, save="Model-Sim", source_season=2045):
    """The fixture season run through the pipeline, then repeated as several seasons so folds and holdouts exist."""
    root = tmp_path / "files"
    shutil.copytree(Path(__file__).parent / "data", root)
    monkeypatch.setattr(paths, "FILES_ROOT", root)
    pipeline.process_season(save, source_season)
    source = paths.output_dir(save, source_season)
    for season in SEASONS:
        target = paths.output_dir(save, season, create=True)
        for file in source.glob("*.csv"):
            shutil.copy(file, target / file.name.replace(str(source_season), str(season)))
        (root / save / "dumps" / "Standard27" / str(season)).mkdir(parents=True, exist_ok=True)
    return root


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """Every spec trained once on the repeated fixture seasons (with cross-validation)."""
    monkeypatch = pytest.MonkeyPatch()
    tmp = tmp_path_factory.mktemp("models")
    root = multi_season_root(tmp, monkeypatch)
    artifacts = {}
    for spec in registry.specs():
        artifacts[spec.name] = registry.train_spec(
            spec, "Model-Sim", "Standard27", SEASONS, cv=True, folds=3, holdout=2
        )
    yield root, artifacts
    monkeypatch.undo()


def test_specs_cover_every_stream_with_the_chosen_families():
    names = {s.name: s for s in registry.specs()}
    assert {"batting-split", "batting", "baserunning", "pitching-fip-split", "pitching-fip"} <= set(names)
    assert {"pitching-babip-split", "pitching-babip"} <= set(names)
    assert {f"fielding-{p}" for p in domains.FIELD_POSITIONS} <= set(names)
    assert names["batting-split"].family == "ridge" and names["pitching-fip-split"].family == "boosting"
    assert names["fielding-6"].family == domains.CHOSEN_FAMILY[domains.FIELDING][6]
    assert len(names) == 15  # three stacked streams x (split + plain), baserunning, eight fielding positions
    assert "batting" not in {s.name for s in registry.specs(include_plain=False)}
    assert [s.name for s in registry.specs(["baserunning"])] == ["baserunning"]
    with pytest.raises(ValueError, match="unknown streams"):
        registry.specs(["bogus"])


@pytest.mark.integration
def test_training_and_scoring_build_the_side_features_the_same_way(tmp_path, monkeypatch):
    multi_season_root(tmp_path, monkeypatch)
    table = data.batting_split_table("Model-Sim", [2046], shared=[])
    ratings = data.read_ratings("Model-Sim", 2046, data.batting_split_rating_columns())
    for side, flag in [(domains.RIGHT, 0.0), (domains.LEFT, 1.0)]:
        expected = data.batting_side_features(ratings, side)
        rows = table.frame[table.frame["side_left"] == flag]
        rebuilt = expected.reindex(rows["ID"])
        assert rebuilt[table.features].to_numpy() == pytest.approx(rows[table.features].to_numpy())
    pitching = data.pitching_split_table("Model-Sim", [2046], "fip", shared=["STM", "PIT"])
    ratings = data.read_ratings("Model-Sim", 2046, data.pitching_split_rating_columns(["STM", "PIT"]))
    rows = pitching.frame[pitching.frame["side_left"] == 1.0]
    rebuilt = data.pitching_side_features(ratings, domains.LEFT, ["STM", "PIT"]).reindex(rows["ID"])
    assert rebuilt[pitching.features].to_numpy() == pytest.approx(rows[pitching.features].to_numpy())


@pytest.mark.integration
def test_every_artifact_records_how_it_was_trained(trained):
    _, artifacts = trained
    assert len(artifacts) == 15
    for loaded in artifacts.values():
        assert loaded.seasons == SEASONS and loaded.save == "Model-Sim" and loaded.ratings == "Standard27"
        assert set(loaded.metrics["cv"]) == {"r2", "mae", "rmse"} and set(loaded.metrics["holdout"]) == {
            "r2",
            "mae",
            "rmse",
        }
        assert loaded.metrics["rows"] > 0 and loaded.notes["stream"] in domains.REVIEWED_FEATURES
    batting = artifacts["batting-split"]
    assert batting.family == "ridge" and batting.notes["kind"] == "split"
    assert set(batting.notes["share_right"]) == {0.0, 1.0, 2.0}  # right, left and switch hitters
    assert set(artifacts["pitching-fip-split"].notes["share_right"]) == {0.0, 1.0}
    assert artifacts["pitching-fip-split"].notes["shared"] == ["STM", "PIT"]
    conversion = artifacts["pitching-babip-split"].notes["conversion"]
    assert 300 < conversion["bip_per_200_ip"] < 900 and 0.5 < conversion["hit_value"] < 1.1


@pytest.mark.integration
def test_scoring_blends_the_two_sides_with_the_platoon_mix_of_the_hand(trained):
    _, artifacts = trained
    players = data.read_output("Model-Sim", 2046, "player-data").drop_duplicates("ID").set_index("ID")
    ratings = players[registry.required_rating_columns(artifacts)].apply(pd.to_numeric, errors="coerce")
    scored = registry.score(artifacts, ratings)
    shares = artifacts["batting-split"].notes["share_right"]
    hand = pd.Series(
        registry.hand_values("batting", data.batting_side_features(ratings, domains.RIGHT)), index=ratings.index
    )
    share = hand.map(shares)
    expected = share * scored["batting_vs_right"] + (1 - share) * scored["batting_vs_left"]
    assert scored["batting"].dropna().to_numpy() == pytest.approx(expected.dropna().to_numpy())
    for column in ["batting_plain", "baserunning", "pitching_fip", "pitching_fip_plain", "pitching_runs"]:
        assert column in scored and scored[column].notna().sum() > 100
    assert {f"fielding_{p}" for p in domains.FIELD_POSITIONS} <= set(scored.columns)
    assert (scored["pitching_runs"] - (scored["pitching_fip"] + scored["pitching_babip_runs"])).abs().max() < 1e-9
    # a rating that is missing gives no prediction for that player and nothing else
    broken = ratings.copy()
    victim = broken.index[0]
    broken.loc[victim, "SPE"] = np.nan
    again = registry.score(artifacts, broken)
    assert np.isnan(again.loc[victim, "baserunning"])
    assert again["baserunning"].drop(victim).dropna().to_numpy() == pytest.approx(
        scored["baserunning"].drop(victim).dropna().to_numpy()
    )


@pytest.mark.integration
def test_models_trained_on_a_dump_save_score_an_export_save(trained, tmp_path, monkeypatch):
    root, artifacts = trained
    pipeline.process_season("TGS27", 2046)
    players = data.read_output("TGS27", 2046, "player-data").drop_duplicates("ID").set_index("ID")
    columns = registry.required_rating_columns(artifacts)
    assert set(columns) <= set(players.columns)
    scored = registry.score(artifacts, players[columns].apply(pd.to_numeric, errors="coerce"))
    assert scored["batting"].notna().sum() > 100 and scored["pitching_runs"].notna().sum() > 100
    assert np.isfinite(scored["batting"].dropna()).all()


@pytest.mark.integration
def test_the_artifacts_survive_saving_and_loading(trained):
    root, artifacts = trained
    for loaded in artifacts.values():
        artifact.save_artifact(loaded)
    folder = artifact.models_dir("Model-Sim", "Standard27")
    assert len(list(folder.glob("*.joblib"))) == 15
    reloaded = artifact.load_artifact("Model-Sim", "Standard27", "pitching-fip-split")
    ratings = data.read_ratings("Model-Sim", 2046, data.pitching_split_rating_columns(["STM", "PIT"])).dropna()
    features = data.pitching_side_features(ratings, domains.RIGHT, ["STM", "PIT"])
    assert reloaded.predict(features) == pytest.approx(artifacts["pitching-fip-split"].predict(features))


@pytest.mark.integration
def test_few_seasons_cap_the_holdout_and_one_season_has_none(tmp_path, monkeypatch):
    multi_season_root(tmp_path, monkeypatch)
    spec = registry.specs(["baserunning"])[0]
    two = registry.train_spec(spec, "Model-Sim", "Standard27", [2046, 2047], folds=3, holdout=5)
    assert two.metrics["holdout_seasons"] == 1 and "holdout" in two.metrics
    one = registry.train_spec(spec, "Model-Sim", "Standard27", [2046], folds=3, holdout=5)
    assert "cv" in one.metrics and "holdout" not in one.metrics

import shutil
from pathlib import Path

import pandas as pd
import pytest

from model import domains
from model.train import FAMILIES
from stats import pipeline
from util import paths


def flat(lists):
    names = []
    for value in lists.values():
        names += [c for v in value.values() for c in v] if isinstance(value, dict) else list(value)
    return names


def test_every_stream_has_a_reviewed_list_and_a_chosen_family():
    assert set(domains.REVIEWED_FEATURES) == {
        domains.BATTING,
        domains.BASERUNNING,
        domains.PITCHING_FIP,
        domains.PITCHING_BABIP,
        domains.FIELDING,
    }
    assert set(domains.CHOSEN_FAMILY) == set(domains.REVIEWED_FEATURES)
    assert set(domains.REVIEWED_FEATURES[domains.FIELDING]) == set(domains.FIELD_POSITIONS)
    assert set(domains.CHOSEN_FAMILY[domains.FIELDING]) == set(domains.FIELD_POSITIONS)
    chosen = [domains.CHOSEN_FAMILY[s] for s in domains.CHOSEN_FAMILY if s != domains.FIELDING]
    chosen += list(domains.CHOSEN_FAMILY[domains.FIELDING].values())
    assert set(chosen) <= set(FAMILIES)


def test_reviewed_lists_hold_no_league_constants_and_no_duplicates():
    for stream, value in domains.REVIEWED_FEATURES.items():
        groups = value.values() if isinstance(value, dict) else [value]
        for features in groups:
            assert len(features) == len(set(features)), stream
            assert not [f for f in features if f.startswith("lg")], stream


def test_the_side_pairs_are_among_the_reviewed_features():
    assert set(c for pair in domains.BATTING_SIDE_PAIRS.values() for c in pair) <= set(
        domains.REVIEWED_FEATURES[domains.BATTING]
    )
    assert set(c for pair in domains.PITCHING_SIDE_PAIRS.values() for c in pair) <= set(
        domains.REVIEWED_FEATURES[domains.PITCHING_FIP]
    )


@pytest.mark.integration
@pytest.mark.parametrize("save, season", [("Model-Sim", 2045), ("TGS27", 2046)])
def test_every_reviewed_rating_exists_in_both_sources(tmp_path, monkeypatch, save, season):
    """The models are trained on dump saves and must be able to score export saves (TGS27) too."""
    root = tmp_path / "files"
    shutil.copytree(Path(__file__).parent / "data", root)
    monkeypatch.setattr(paths, "FILES_ROOT", root)
    pipeline.process_season(save, season)
    ratings = pd.read_csv(paths.output_file(save, season, "player-data"), low_memory=False)
    needed = set(flat(domains.REVIEWED_FEATURES)) | {"B", "T"}
    assert needed <= set(ratings.columns), sorted(needed - set(ratings.columns))

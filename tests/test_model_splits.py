import numpy as np
import pytest

from model import splits


def groups(n_players=40, seasons=6, seed=0):
    rng = np.random.default_rng(seed)
    ids = np.repeat(np.arange(1, n_players + 1), seasons)
    return rng.permutation(ids)


def test_every_row_is_in_exactly_one_fold_and_a_player_stays_in_one_fold():
    g = groups()
    folds = splits.group_folds(g, n_splits=5, seed=1)
    assert set(folds) == set(range(5))
    for player in np.unique(g):
        assert len(set(folds[g == player])) == 1
    sizes = [len(np.unique(g[folds == k])) for k in range(5)]
    assert max(sizes) - min(sizes) <= 1  # balanced by player


def test_group_kfold_never_puts_a_player_on_both_sides():
    g = groups()
    seen = []
    for train, test in splits.group_kfold(g, n_splits=4, seed=2):
        assert not set(g[train]) & set(g[test])
        assert len(train) + len(test) == len(g)
        seen.extend(test)
    assert sorted(seen) == list(range(len(g)))


def test_the_same_seed_gives_the_same_split_and_another_seed_a_different_one():
    g = groups()
    assert (splits.group_folds(g, 5, seed=3) == splits.group_folds(g, 5, seed=3)).all()
    assert not (splits.group_folds(g, 5, seed=3) == splits.group_folds(g, 5, seed=4)).all()


def test_too_many_folds_are_rejected():
    with pytest.raises(ValueError):
        splits.group_folds([1, 1, 2, 2], n_splits=3)
    with pytest.raises(ValueError):
        splits.group_folds([1, 2, 3], n_splits=1)


def test_last_seasons_holdout():
    seasons = np.repeat(np.arange(2046, 2056), 3)
    train, test = splits.last_seasons_holdout(seasons, 2)
    assert set(seasons[test]) == {2054, 2055}
    assert seasons[train].max() == 2053
    assert len(train) + len(test) == len(seasons)
    with pytest.raises(ValueError):
        splits.last_seasons_holdout(seasons, 10)


def test_player_holdout_keeps_players_apart_and_is_deterministic():
    g = groups()
    train, test = splits.player_holdout(g, 0.25, seed=5)
    assert not set(g[train]) & set(g[test])
    assert len(np.unique(g[test])) == 10
    assert (test == splits.player_holdout(g, 0.25, seed=5)[1]).all()

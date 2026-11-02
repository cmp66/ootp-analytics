"""Train / validation splits that never put one player on both sides.

A player appears in about six seasons and his ratings change slowly, so a random row split lets the model
memorize players. Everything here takes the player ids (groups) and a seed and is deterministic."""

import numpy as np


def group_folds(groups, n_splits: int = 5, seed: int = 0) -> np.ndarray:
    """Fold number (0 to n_splits - 1) of every row; all rows of a group share a fold, folds are balanced by group."""
    groups = np.asarray(groups)
    unique = np.unique(groups)
    if n_splits < 2 or n_splits > len(unique):
        raise ValueError(f"n_splits must be between 2 and the number of groups ({len(unique)}), got {n_splits}")
    order = np.random.default_rng(seed).permutation(len(unique))
    fold_of_group = np.empty(len(unique), dtype=int)
    fold_of_group[order] = np.arange(len(unique)) % n_splits
    return fold_of_group[np.searchsorted(unique, groups)]


def group_kfold(groups, n_splits: int = 5, seed: int = 0):
    """Yield (train_index, validation_index) for each fold."""
    folds = group_folds(groups, n_splits, seed)
    for fold in range(n_splits):
        yield np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)


def last_seasons_holdout(seasons, n_seasons: int):
    """(train_index, test_index): the last n_seasons seasons are the test set, as when predicting a future season."""
    seasons = np.asarray(seasons)
    if n_seasons < 1 or n_seasons >= len(np.unique(seasons)):
        raise ValueError("n_seasons must be at least 1 and below the number of seasons")
    cutoff = np.sort(np.unique(seasons))[-n_seasons]
    return np.flatnonzero(seasons < cutoff), np.flatnonzero(seasons >= cutoff)


def player_holdout(groups, test_fraction: float = 0.2, seed: int = 0):
    """(train_index, test_index) with a fraction of the groups (players) held out entirely."""
    groups = np.asarray(groups)
    unique = np.unique(groups)
    n_test = max(1, int(round(len(unique) * test_fraction)))
    test_groups = np.random.default_rng(seed).choice(unique, size=n_test, replace=False)
    in_test = np.isin(groups, test_groups)
    return np.flatnonzero(~in_test), np.flatnonzero(in_test)

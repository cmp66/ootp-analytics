"""Comparisons used to pick the model family and the platoon design for each value stream (plans/14-models.md).

The folds are fixed per player, from one assignment for all the tables of a stream, so a plain model and a stacked
platoon model are scored on the same players and the same rows."""

import numpy as np
import pandas as pd

from model import splits
from model.metrics import summarize
from model.train import fit_table, make_model

FAMILY_ORDER = ["ridge", "boosting", "mlp"]  # simplest first


def choose_family(scores: dict[str, float], tolerance: float = 0.003) -> str:
    """The simplest family whose score is within `tolerance` of the best (higher is better)."""
    best = max(scores.values())
    for family in FAMILY_ORDER:
        if family in scores and scores[family] >= best - tolerance:
            return family
    raise ValueError("no known family in scores")


def player_folds(ids, n_splits: int = 5, seed: int = 0) -> dict:
    """player id -> fold, the same for every table that uses it."""
    unique = np.unique(np.asarray(ids))
    return dict(zip(unique, splits.group_folds(unique, n_splits, seed)))


def cross_validate_folds(family: str, table, folds: dict, n_splits: int = 5, seed: int = 0, **params) -> np.ndarray:
    """Out-of-fold predictions of a plain table with the given player folds."""
    fold_of_row = table.frame[table.groups].map(folds).to_numpy()
    prediction = np.full(len(table), np.nan)
    for fold in range(n_splits):
        train, test = np.flatnonzero(fold_of_row != fold), np.flatnonzero(fold_of_row == fold)
        if len(test) == 0:
            continue
        fitted = fit_table(make_model(family, **params), table, train, seed)
        prediction[test] = fitted.predict(table.frame.iloc[test][table.features])
    return prediction


def platoon_mix(split_frame: pd.DataFrame, hand_column: str | None = None) -> dict:
    """Share of the volume that is against right-handed opponents, overall or per hand value.

    Uses the weights of the stacked table (PA against each side, or IP): the league's real mix, not an assumption."""
    right = split_frame[split_frame["side_left"] == 0]
    left = split_frame[split_frame["side_left"] == 1]
    if hand_column is None:
        return {None: right["weight"].sum() / (right["weight"].sum() + left["weight"].sum())}
    r = right.groupby(hand_column)["weight"].sum()
    l_ = left.groupby(hand_column)["weight"].sum()
    return (r / (r + l_)).to_dict()


def blend_sides(split_frame: pd.DataFrame, prediction, share_right, hand=None) -> pd.DataFrame:
    """One row per player-season with the prediction against each side and the platoon-weighted total.

    share_right is a number, or a dict by hand value with `hand` the per-row hand values. Player-seasons that lack
    one of the sides are left out."""
    frame = pd.DataFrame(
        {
            "ID": split_frame["ID"].to_numpy(),
            "season": split_frame["season"].to_numpy(),
            "side": np.where(split_frame["side_left"].to_numpy() == 1, "left", "right"),
            "prediction": np.asarray(prediction),
        }
    )
    if hand is not None:
        frame["hand"] = np.asarray(hand)
    wide = frame.pivot_table(index=["ID", "season"], columns="side", values="prediction").dropna()
    if hand is None:
        share = share_right if not isinstance(share_right, dict) else share_right[None]
    else:
        hands = frame.drop_duplicates(["ID", "season"]).set_index(["ID", "season"])["hand"].reindex(wide.index)
        share = hands.map(share_right).to_numpy()
    wide["total"] = share * wide["right"] + (1 - share) * wide["left"]
    return wide.reset_index()


def cross_validate_stacked(
    family: str,
    split_table,
    total_table,
    folds: dict,
    share_right,
    hand_column: str | None = None,
    n_splits: int = 5,
    seed: int = 0,
    **params,
):
    """Score a stacked platoon model against the *total* target of the plain table.

    The stacked model is trained on rows of players outside the fold and predicts both sides of each held-out
    player-season; the two are blended with the league's platoon mix. Returns (rows, metrics) with one row per
    player-season present in both tables; the plain table's own predictions are not involved."""
    fold_of_row = split_table.frame[split_table.groups].map(folds).to_numpy()
    out = []
    for fold in range(n_splits):
        train, test = np.flatnonzero(fold_of_row != fold), np.flatnonzero(fold_of_row == fold)
        if len(test) == 0:
            continue
        fitted = fit_table(make_model(family, **params), split_table, train, seed)
        rows = split_table.frame.iloc[test]
        hand = None if hand_column is None else rows[hand_column]
        out.append(blend_sides(rows, fitted.predict(rows[split_table.features]), share_right, hand))
    blended = pd.concat(out, ignore_index=True)
    key = [total_table.groups, total_table.season]
    truth = total_table.frame[[*key, total_table.target, total_table.weight]].rename(
        columns={total_table.groups: "ID", total_table.season: "season"}
    )
    merged = blended.merge(truth, on=["ID", "season"])
    return merged, summarize(merged[total_table.target], merged["total"], merged[total_table.weight])


def babip_to_runs(babip_difference, bip_per_200_ip: float, hit_value: float) -> np.ndarray:
    """A predicted BABIP difference (above the league) as runs saved per 200 innings: fewer hits on balls in play
    mean fewer runs, so the sign flips. bip_per_200_ip is the league's balls in play per 200 innings."""
    return -np.asarray(babip_difference, dtype=float) * bip_per_200_ip * hit_value

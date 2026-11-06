"""League-relative predicted values: a second step after the raw predictions.

The fielding targets (and the batting ones) are values against the league average, so a model trained on one league
predicts values against that league's average. In a league that selects for defence (human managers who play good
defenders at shortstop, center field, second and third base) the same player is worth less against his own league's
average. The step here shifts every prediction column by the playing-time weighted mean of that column over the players
of the scored season who played (the same volume minimums as in training), written to new `<column>_rel` columns. The
raw columns are not changed.
"""

import numpy as np
import pandas as pd

from model import data, domains

# raw-BABIP columns are in BABIP units and are only used through pitching_babip_runs
NOT_SHIFTED = {"pitching_babip", "pitching_babip_vs_right", "pitching_babip_vs_left", "pitching_babip_plain"}


def season_weights(save: str, season: int) -> dict[str, pd.Series]:
    """Playing time of the scored season by player ID for each family of prediction columns."""
    hitting = data.read_output(save, season, "hitting")
    hitting = hitting[hitting["PA"] >= domains.MIN_PA]
    pitching = data.read_output(save, season, "pitching")
    fielding = data.read_output(save, season, "fielding")
    fielding = fielding[fielding["IPClean"] >= domains.MIN_FIELDING_IP]
    weights = {
        domains.BATTING: hitting.groupby("ID")["PA"].sum().astype(float),
        domains.BASERUNNING: hitting.groupby("ID")["PA"].sum().astype(float),
        domains.PITCHING_FIP: pitching[pitching["IPClean"] >= domains.MIN_IP].groupby("ID")["IPClean"].sum(),
        domains.PITCHING_BABIP: pitching[pitching["BIP"] >= domains.MIN_BIP].groupby("ID")["BIP"].sum().astype(float),
    }
    for position in domains.FIELD_POSITIONS:
        rows = fielding[fielding["POS"] == position]
        weights[f"fielding_{position}"] = rows.groupby("ID")["IPClean"].sum()
    return weights


def family_of(column: str) -> str | None:
    """The weights family a prediction column belongs to (None for columns that are not shifted)."""
    if column in NOT_SHIFTED:
        return None
    if column.startswith("pitching_fip"):
        return domains.PITCHING_FIP
    if column.startswith("pitching_babip"):  # pitching_babip_runs
        return domains.PITCHING_BABIP
    for family in (domains.BATTING, domains.BASERUNNING):
        if column.startswith(family):
            return family
    if column.startswith("fielding_"):
        return "_".join(column.split("_")[:2])
    return None


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    """Mean of `values` (indexed by ID) weighted by playing time; NaN when nobody has both."""
    both = pd.concat([values, weights.rename("weight")], axis=1, join="inner").dropna()
    both = both[both["weight"] > 0]
    if both.empty:
        return float("nan")
    return float((both.iloc[:, 0] * both["weight"]).sum() / both["weight"].sum())


def add_relative(scored: pd.DataFrame, weights: dict[str, pd.Series]) -> tuple[pd.DataFrame, dict[str, float]]:
    """Add `<column>_rel` for every shifted column; also returns the shift (league reference) of each column.

    pitching_runs is the sum of its two parts, so its shift is the sum of their shifts."""
    out = scored.copy()
    shifts: dict[str, float] = {}
    for column in scored.columns:
        family = family_of(column)
        if family is None or column == "pitching_runs":
            continue
        shifts[column] = weighted_mean(scored[column], weights[family])
    if "pitching_runs" in scored:
        shifts["pitching_runs"] = shifts.get("pitching_fip", float("nan")) + shifts.get(
            "pitching_babip_runs", float("nan")
        )
    for column, shift in shifts.items():
        out[f"{column}_rel"] = scored[column] - shift
    return out, shifts


CALIBRATED_RATINGS = {  # calibration name -> the rating columns (the mean of the vs-right and vs-left ratings)
    "control": ("CON.1 vR", "CON.1 vL"),
    "pbabip": ("PBABIP vR", "PBABIP vL"),
    "stamina": ("STM",),
}
MAX_Z = 2.5  # the calibration was measured among major leaguers; a minor leaguer further out is not extrapolated
CALIBRATED_FIP = ("pitching_fip", "pitching_fip_vs_right", "pitching_fip_vs_left", "pitching_fip_plain")


def rating_z(players: pd.DataFrame, weights: pd.Series, columns: tuple[str, ...]) -> pd.Series:
    """Each pitcher's rating (the mean of the columns, e.g. the vs-right and vs-left ratings) in standard deviations from the
    league: the mean and standard deviation are those of the pitchers who pitched in the season, weighted by innings."""
    rating = players[list(columns)].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    both = pd.concat([rating.rename("rating"), weights.rename("weight")], axis=1, join="inner").dropna()
    both = both[both["weight"] > 0]
    if both.empty:
        return pd.Series(np.nan, index=players.index)
    mean = np.average(both["rating"], weights=both["weight"])
    sd = np.sqrt(np.average((both["rating"] - mean) ** 2, weights=both["weight"]))
    return (rating - mean) / sd if sd > 0 else pd.Series(0.0, index=players.index)


def control_z(players: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """The control rating in standard deviations of the league (see rating_z)."""
    return rating_z(players, weights, CALIBRATED_RATINGS["control"])


def calibrate_pitching(
    scored: pd.DataFrame, players: pd.DataFrame, weights: dict[str, pd.Series], per_sd: dict[str, float] | float
):
    """Human league calibration of the pitching values: runs per 200 IP for each standard deviation of a rating, for the
    ratings of CALIBRATED_RATINGS (`per_sd` maps the name to the number; a plain number is control alone). The models learn from
    a computer league, and in the human leagues pitchers with poor control, hit prevention (PBABIP) or stamina do worse, and those
    with good ones better, than they predict (about 2.4, 1.1 and 1.2 runs per 200 IP per standard deviation, pooled over three
    leagues, scripts/check_human_bias.py). Added to the FIP values and to pitching_runs, with each amount in
    `pitching_<name>_adjustment` (the standard deviations are capped at +-MAX_Z); a setting of 0 changes nothing."""
    if not isinstance(per_sd, dict):
        per_sd = {"control": per_sd}
    per_sd = {name: value for name, value in per_sd.items() if value}
    if not per_sd or "pitching_fip" not in scored:
        return scored
    out = scored.copy()
    total = pd.Series(0.0, index=scored.index)
    for name, value in per_sd.items():
        z = rating_z(players, weights[domains.PITCHING_FIP], CALIBRATED_RATINGS[name]).clip(-MAX_Z, MAX_Z)
        adjustment = (value * z).reindex(scored.index).fillna(0.0)
        out[f"pitching_{name}_adjustment"] = adjustment
        total += adjustment
    for column in (*CALIBRATED_FIP, "pitching_runs"):
        if column in out:
            out[column] = out[column] + total
    return out

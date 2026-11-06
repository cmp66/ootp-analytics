"""Check whether the pitching models are too kind to pitchers who have stuff and little else (or too hard on the opposite kind).

Usage:
    PYTHONPATH=. python scripts/check_stuff_bias.py Model-Sim                       # every season with data
    PYTHONPATH=. python scripts/check_stuff_bias.py Model-Sim --start 2081 --stop 2086
    PYTHONPATH=. python scripts/check_stuff_bias.py Model-Sim --stuff-min 70 --others-max 45   # stricter groups
    PYTHONPATH=. python scripts/check_stuff_bias.py TGS27 --models-from Model-Sim    # another save's outcomes, same ratings set

Needs a save with outcomes (a training league: Model-Sim, Model-Sim-26-Std) and its trained models. For each pitching model
(`pitching-fip`: park adjusted FIP runs SAVED per 200 IP against the average, so more is better for the pitcher;
`pitching-babip`: BABIP minus the league's, so less is better) the player-seasons are put in groups by their ratings (the mean of
the vs-right and vs-left ratings):

  stuff only      stuff at least --stuff-min and control, home run avoidance and movement (PBABIP) all at most --others-max
  control only    the mirror: control and home run avoidance both at least --stuff-min, stuff at most --others-max
  balanced        the four ratings within --gap points of each other
  all             everybody

and, for each group, the mean prediction, the mean outcome (weighted by innings or balls in play) and the difference. A group
whose pitchers did clearly worse than predicted (fewer runs saved, more hits allowed) is one the model is too kind to, and one
that did better is one it is too harsh to; z is the difference over its standard error (rows are treated as independent,
though a pitcher appears in several seasons, so read |z| above 3 as a sign, not a proof). Starters (start share of at least
half) and relievers are shown apart.

This checks the value now. The ceiling (potential) models build on the same ratings but on development tables, so a bias
here is a bias there too, but a clean result does not clear the ceiling.
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from model import artifact, domains, registry, scales  # noqa: E402
from util.config import get_save  # noqa: E402
from util.seasons import available_seasons, select_seasons  # noqa: E402

PARTS = {  # model name -> (stream, what the target is, unit)
    "pitching-fip": (domains.PITCHING_FIP, "FIP runs saved per 200 IP against the average", "runs"),
    "pitching-babip": (domains.PITCHING_BABIP, "BABIP above the league's", "BABIP"),
}
MIN_ROWS = 30  # a group with fewer player-seasons is shown but not judged


@dataclass(frozen=True)
class Groups:
    stuff_min: float = 65.0
    others_max: float = 50.0
    gap: float = 10.0


def profile(frame: pd.DataFrame) -> pd.DataFrame:
    """The four ratings of a pitcher (mean of vs-right and vs-left) and the weakest and strongest of the other three."""
    mean = lambda a, b: (frame[a].astype(float) + frame[b].astype(float)) / 2  # noqa: E731
    out = pd.DataFrame(
        {
            "stuff": mean("STU vR", "STU vL"),
            "control": mean("CON.1 vR", "CON.1 vL"),
            "homers": mean("HRR vR", "HRR vL"),
            "movement": mean("PBABIP vR", "PBABIP vL"),
        },
        index=frame.index,
    )
    out["others_max"] = out[["control", "homers", "movement"]].max(axis=1)
    out["spread"] = out[["stuff", "control", "homers", "movement"]].max(axis=1) - out[
        ["stuff", "control", "homers", "movement"]
    ].min(axis=1)
    return out


def group_masks(ratings: pd.DataFrame, groups: Groups) -> dict[str, pd.Series]:
    return {
        "stuff only": (ratings["stuff"] >= groups.stuff_min) & (ratings["others_max"] <= groups.others_max),
        "control only": (ratings[["control", "homers"]].min(axis=1) >= groups.stuff_min)
        & (ratings["stuff"] <= groups.others_max),
        "balanced": ratings["spread"] <= groups.gap,
        "all": pd.Series(True, index=ratings.index),
    }


def compare(y, prediction, weight) -> dict:
    """Mean prediction, mean outcome and their difference (outcome minus prediction) with its standard error, weighted."""
    y, p, w = (np.asarray(a, dtype=float) for a in (y, prediction, weight))
    n = len(y)
    if n == 0 or w.sum() <= 0:
        return {
            "n": n,
            "weight": 0.0,
            "prediction": np.nan,
            "outcome": np.nan,
            "diff": np.nan,
            "se": np.nan,
            "z": np.nan,
        }
    error = y - p
    diff = float(np.average(error, weights=w))
    se = float(np.sqrt(np.sum(w**2 * (error - diff) ** 2)) / w.sum())
    return {
        "n": n, "weight": float(w.sum()), "prediction": float(np.average(p, weights=w)),
        "outcome": float(np.average(y, weights=w)), "diff": diff, "se": se,
        "z": diff / se if se > 0 else np.nan,
    }  # fmt: skip


def report(
    frame: pd.DataFrame, prediction, ratings: pd.DataFrame, roles: pd.Series, groups: Groups, unit: str
) -> list[str]:
    """Lines of the table for one model: a row per group and role."""
    masks = group_masks(ratings, groups)
    prediction = pd.Series(np.asarray(prediction, dtype=float), index=frame.index)
    lines = [
        f"    {'group':<14}{'role':<10}{'rows':>7}{'weight':>9}{'predicted':>11}{'actual':>9}{'actual-pred':>13}{'z':>7}"
    ]
    for name, mask in masks.items():
        for role in ("starters", "relievers", "both"):
            keep = mask if role == "both" else mask & (roles == ("starter" if role == "starters" else "reliever"))
            c = compare(frame.loc[keep, "target"], prediction[keep], frame.loc[keep, "weight"])
            if c["n"] == 0:
                continue
            note = "" if c["n"] >= MIN_ROWS else "  (few rows)"
            lines.append(
                f"    {name:<14}{role:<10}{c['n']:>7}{c['weight']:>9.0f}{c['prediction']:>+11.3f}{c['outcome']:>+9.3f}"
                f"{c['diff']:>+13.3f}{c['z']:>+7.1f}{note}"
            )
    if unit == "runs":  # FIPRuns200 counts runs SAVED against the league rate: more is better for the pitcher
        note = "runs saved: negative actual-pred = fewer runs saved than expected = too kind; positive = too harsh"
    else:
        note = f"{unit}: positive actual-pred = more allowed than the model expected = the model is too kind"
    lines.append(f"    ({note})")
    return lines


def roles_of(frame: pd.DataFrame) -> pd.Series:
    share = frame["start_share"].astype(float) if "start_share" in frame else pd.Series(np.nan, index=frame.index)
    return pd.Series(np.where(share >= 0.5, "starter", "reliever"), index=frame.index)


def check(save: str, models_from: str, seasons: list[int], groups: Groups) -> list[str]:
    """The report for every pitching model of the models' set, scored on the outcomes of `save`."""
    ratings_set = get_save(models_from).ratings
    model_set = scales.model_set(ratings_set, get_save(save).scales)
    fip_table = registry.build_table(
        registry.Spec("pitching-fip", domains.PITCHING_FIP, "plain", "ridge"), save, seasons
    )
    profiles = profile(fip_table.frame)
    keyed = fip_table.frame[["ID", "season"]].join(profiles).join(roles_of(fip_table.frame).rename("role"))
    out = []
    for name, (stream, what, unit) in PARTS.items():
        loaded = artifact.load_artifact(models_from, model_set, name)
        table = registry.build_table(registry.Spec(name, stream, "plain", loaded.family), save, seasons)
        frame = table.frame
        if name != "pitching-fip":  # the ratings and the role come from the FIP table of the same player-seasons
            frame = frame.merge(keyed, on=["ID", "season"], how="inner")
            ratings, roles = frame[profiles.columns], frame["role"]
        else:
            ratings, roles = profiles, keyed["role"]
        prediction = loaded.predict(frame[loaded.features])
        out.append(f"{name}: {what}, {len(frame)} player-seasons of {save} {seasons[0]}-{seasons[-1]}")
        out += report(frame, prediction, ratings, roles, groups, unit)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("save")
    parser.add_argument("--start", type=int, help="first season (default: the first with data)")
    parser.add_argument("--stop", type=int, help="last season (default: the latest)")
    parser.add_argument("--models-from", help="save whose models to check (default: the same save; same ratings set)")
    parser.add_argument(
        "--stuff-min", type=float, default=Groups.stuff_min, help="stuff (or control and home runs) at least"
    )
    parser.add_argument("--others-max", type=float, default=Groups.others_max, help="the other ratings at most")
    parser.add_argument(
        "--gap", type=float, default=Groups.gap, help="balanced: the four ratings within this many points"
    )
    args = parser.parse_args(argv)
    try:
        available = available_seasons(args.save)
        seasons = select_seasons(
            available, args.start if args.start is not None else available[0], args.stop or available[-1]
        )
        models_from = args.models_from or args.save
        if get_save(models_from).ratings != get_save(args.save).ratings:
            raise ValueError(f"{models_from} models are for ratings {get_save(models_from).ratings!r}")
        lines = check(args.save, models_from, seasons, Groups(args.stuff_min, args.others_max, args.gap))
    except (KeyError, ValueError, FileNotFoundError, IndexError) as error:
        print(f"error: {error.args[0] if isinstance(error, KeyError) else error}", file=sys.stderr)
        return 2
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())

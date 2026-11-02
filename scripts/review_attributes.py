"""Stage 3 of plans/14-models.md: which ratings belong in each model?

For every value stream a core list of ratings is compared with
  - add-one: core + one candidate (or a group, such as the batter-hand indicators),
  - drop-one: core minus one rating,
  - alternative sets (overall ratings versus the vs-right / vs-left pair, the blend of the pair, ...),
using player-grouped cross-validation repeated over seeds, on the same folds for every configuration, so each
difference is paired. A change counts when the mean fold difference in weighted R^2 is at least MIN_GAIN and at
least 3 standard errors from zero. Pitcher BABIP, which is mostly noise, is judged by a regression with
player-clustered standard errors instead.

Usage: PYTHONPATH=. python scripts/review_attributes.py [save] [--streams batting baserunning pitching-fip
       pitching-babip fielding] [--start 2046 --stop 2086] [--seeds 2] [--folds 5] [--out plans/data/attribute-review]
"""

import argparse
import logging
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from model import data, domains, splits
from model.metrics import weighted_r2
from model.train import fit_table, make_model

MIN_GAIN = 0.002
MIN_T = 3.0

LEAGUE_BATTING = ["lgBabipRating", "lgAvkRating", "lgConRating", "lgEyeRating", "lgPowRating"]
LEAGUE_PITCHING = ["lgPbabipRating", "lgStuRating", "lgHrrRating", "lgCtlRating", "lgMovRating"]

BATTING_CORE = ["BA vR", "BA vL", "GAP vR", "GAP vL", "POW vR", "POW vL", "EYE vR", "EYE vL", "K vR", "K vL"]
BATTING_CANDIDATES = {
    "bats_left+bats_switch": ["bats_left", "bats_switch"],
    "batting hit-by-pitch rating": ["batting_ratings_overall_hp"],
    "GBT (ground ball type)": ["GBT"],
    "FBT (fly ball type)": ["FBT"],
    "BUN (bunt)": ["BUN"],
    "BFH (bunt for hit)": ["BFH"],
    "SPE (speed)": ["SPE"],
    "Age": ["Age"],
    "HT": ["HT"],
    "WT": ["WT"],
    "CON (derived: BABIP + K's)": ["CON"],
    "league batting constants": LEAGUE_BATTING,
}
BATTING_SKILLS = {f"{n} (vR and vL)": list(cols) for n, cols in domains.BATTING_SIDE_PAIRS.items()}
BATTING_SETS = {
    "overall ratings (BABIP, GAP, POW, EYE, K's)": ["BABIP", "GAP", "POW", "EYE", "K's"],
    "vR / vL pair (core)": BATTING_CORE,
    "0.73 vR + 0.27 vL blend (5)": ["w_BA", "w_GAP", "w_POW", "w_EYE", "w_K"],
    "pair + batter hand": BATTING_CORE + ["bats_left", "bats_switch"],
    "old list (pair + league constants)": BATTING_CORE + LEAGUE_BATTING,
}

BASERUNNING_CORE = ["SPE", "SR", "STE", "RUN"]
BASERUNNING_CANDIDATES = {
    "Age": ["Age"],
    "WT": ["WT"],
    "HT": ["HT"],
    "GBT": ["GBT"],
    "BUN": ["BUN"],
    "POW": ["POW vR"],
    "league batting constants": LEAGUE_BATTING,
}
BASERUNNING_SETS = {"SPE+STE+RUN only": ["SPE", "STE", "RUN"], "SR+STE only": ["SR", "STE"]}

PITCH_CORE = ["STU vR", "STU vL", "CON.1 vR", "CON.1 vL", "PBABIP vR", "PBABIP vL", "HRR vR", "HRR vL"]
PITCH_CANDIDATES = {
    "T (throws)": ["T"],
    "STM (stamina)": ["STM"],
    "HLD (holds runners)": ["HLD"],
    "PIT (number of pitches)": ["PIT"],
    "G/F (ground / fly)": ["G/F"],
    "VELO": ["VELO"],
    "Slot": ["Slot"],
    "Age": ["Age"],
    "HT": ["HT"],
    "WT": ["WT"],
    "MOV (derived: HRR + PBABIP)": ["MOV"],
    "pitching hit-by-pitch rating": ["pitching_ratings_overall_hp"],
    "wild pitch rating": ["pitching_ratings_overall_wild_pitch"],
    "league pitching constants": LEAGUE_PITCHING,
}
PITCH_SKILLS = {f"{n} (vR and vL)": list(cols) for n, cols in domains.PITCHING_SIDE_PAIRS.items()}
PITCH_SETS = {
    "overall ratings (STU, CON.1, PBABIP, HRR)": ["STU", "CON.1", "PBABIP", "HRR"],
    "vR / vL pair (core)": PITCH_CORE,
    "old list (pair + T, STM, HLD, PIT + league constants)": PITCH_CORE + ["T", "STM", "HLD", "PIT"] + LEAGUE_PITCHING,
}

BABIP_CANDIDATES = [
    "PBABIP vR",
    "PBABIP vL",
    "STU vR",
    "STU vL",
    "HRR vR",
    "HRR vL",
    "CON.1 vR",
    "CON.1 vL",
    "G/F",
    "MOV",
    "HT",
    "WT",
]

FIELD_CORE = {
    2: ["C ABI", "C ARM", "C FRM"],
    3: ["IF RNG", "IF ARM", "TDP", "IF ERR"],
    4: ["IF RNG", "IF ARM", "TDP", "IF ERR"],
    5: ["IF RNG", "IF ARM", "TDP", "IF ERR"],
    6: ["IF RNG", "IF ARM", "TDP", "IF ERR"],
    7: ["OF RNG", "OF ARM", "OF ERR"],
    8: ["OF RNG", "OF ARM", "OF ERR"],
    9: ["OF RNG", "OF ARM", "OF ERR"],
}
POSITION_RATING = {2: "C", 3: "1B", 4: "2B", 5: "3B", 6: "SS", 7: "LF", 8: "CF", 9: "RF"}
LEAGUE_FIELDING = {
    2: ["lgCBlock", "lgCArm", "lgCFrm"],
    3: ["lg1BArm", "lg1BError", "lg1BRange"],
    4: ["lg2BArm", "lg2BError", "lg2BRange"],
    5: ["lg3BArm", "lg3BError", "lg3BRange"],
    6: ["lgSSArm", "lgSSError", "lgSSRange"],
    7: ["lgLFArm", "lgLFError", "lgLFRange"],
    8: ["lgCFArm", "lgCFError", "lgCFRange"],
    9: ["lgRFArm", "lgRFError", "lgRFRange"],
}


def field_candidates(position: int) -> dict:
    return {
        "SPE (speed)": ["SPE"],
        "WT": ["WT"],
        "HT": ["HT"],
        "Age": ["Age"],
        "composite position rating": [POSITION_RATING[position]],
        "pitcher league BABIP rating (plan 10: no effect expected)": ["lgPbabipRating"],
        "league fielding constants": LEAGUE_FIELDING[position],
    }


def fold_scores(family: str, table, features: list[str], seeds: int, folds: int) -> np.ndarray:
    """Weighted R^2 of every fold of every seed; the folds are the same for every feature list."""
    sub = replace(table, features=list(features))
    groups = table.frame[table.groups].to_numpy()
    scores = []
    for seed in range(seeds):
        for train, test in splits.group_kfold(groups, folds, seed):
            fitted = fit_table(make_model(family), sub, train, seed)
            rows = table.frame.iloc[test]
            scores.append(weighted_r2(rows[table.target], fitted.predict(rows[features]), rows[table.weight]))
    return np.array(scores)


def compare(base: np.ndarray, other: np.ndarray) -> tuple[float, float]:
    """Mean paired fold difference (other - base) and its standard error."""
    diff = other - base
    return float(diff.mean()), float(diff.std(ddof=1) / np.sqrt(len(diff)))


def verdict_add(delta: float, se: float) -> str:
    if delta >= MIN_GAIN and delta / max(se, 1e-9) >= MIN_T:
        return "ADD"
    if delta <= -MIN_GAIN and delta / max(se, 1e-9) <= -MIN_T:
        return "HURTS"
    return "no gain"


def verdict_drop(delta: float, se: float) -> str:
    """delta = R^2 without the rating minus with it."""
    if delta <= -MIN_GAIN and delta / max(se, 1e-9) <= -MIN_T:
        return "KEEP"
    return "can drop"


def review_stream(
    label: str, table, core: list[str], candidates: dict, sets: dict, family: str, seeds: int, folds: int, groups=None
):
    started = time.perf_counter()
    base = fold_scores(family, table, core, seeds, folds)
    rows = [
        {
            "stream": label,
            "test": "core",
            "item": ", ".join(core),
            "family": family,
            "r2": base.mean(),
            "delta": 0.0,
            "se": 0.0,
            "verdict": "",
        }
    ]
    for name, columns in candidates.items():
        delta, se = compare(
            base, fold_scores(family, table, core + [c for c in columns if c not in core], seeds, folds)
        )
        rows.append(
            {
                "stream": label,
                "test": "add",
                "item": name,
                "family": family,
                "r2": base.mean() + delta,
                "delta": delta,
                "se": se,
                "verdict": verdict_add(delta, se),
            }
        )
    if len(core) > 1:
        for column in core:
            kept = [c for c in core if c != column]
            delta, se = compare(base, fold_scores(family, table, kept, seeds, folds))
            rows.append(
                {
                    "stream": label,
                    "test": "drop",
                    "item": column,
                    "family": family,
                    "r2": base.mean() + delta,
                    "delta": delta,
                    "se": se,
                    "verdict": verdict_drop(delta, se),
                }
            )
    for name, columns in (groups or {}).items():
        kept = [c for c in core if c not in columns]
        delta, se = compare(base, fold_scores(family, table, kept, seeds, folds))
        rows.append(
            {
                "stream": label,
                "test": "drop skill",
                "item": name,
                "family": family,
                "r2": base.mean() + delta,
                "delta": delta,
                "se": se,
                "verdict": verdict_drop(delta, se),
            }
        )
    for name, columns in sets.items():
        delta, se = compare(base, fold_scores(family, table, columns, seeds, folds))
        rows.append(
            {
                "stream": label,
                "test": "set",
                "item": name,
                "family": family,
                "r2": base.mean() + delta,
                "delta": delta,
                "se": se,
                "verdict": verdict_add(delta, se),
            }
        )
    print(f"{label}: {len(table)} rows, {family}, {time.perf_counter() - started:.0f}s", flush=True)
    return rows


def add_helper_columns(table):
    frame = table.frame
    if "B" in frame:
        frame["bats_left"] = (frame["B"] == 2).astype(float)
        frame["bats_switch"] = (frame["B"] == 3).astype(float)
    for name, (right, left) in domains.BATTING_SIDE_PAIRS.items():
        if right in frame and left in frame:
            frame[f"w_{name}"] = 0.73 * frame[right] + 0.27 * frame[left]


def clustered_regression(table, columns: list[str]) -> pd.DataFrame:
    """Weighted least squares of the target on the ratings with standard errors clustered by player."""
    frame = table.frame
    X = np.column_stack([np.ones(len(frame)), frame[columns].to_numpy(dtype=float)])
    w = frame[table.weight].to_numpy()
    sw = np.sqrt(w)[:, None]
    y = frame[table.target].to_numpy()
    beta = np.linalg.lstsq(X * sw, y * sw[:, 0], rcond=None)[0]
    resid = (y - X @ beta) * w
    bread = np.linalg.inv((X * w[:, None]).T @ X)
    meat = np.zeros((X.shape[1], X.shape[1]))
    for rows in frame.groupby(table.groups).indices.values():
        score = X[rows].T @ resid[rows]
        meat += np.outer(score, score)
    se = np.sqrt(np.diag(bread @ meat @ bread))
    sd = X[:, 1:].std(axis=0)
    return pd.DataFrame(
        {"rating": columns, "per_rating_point": beta[1:], "t": beta[1:] / se[1:], "effect_per_sd": beta[1:] * sd}
    )


def main() -> None:
    logging.getLogger("model.data").setLevel(logging.ERROR)  # the tables report dropped rows; not needed here
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("save", nargs="?", default="Model-Sim")
    parser.add_argument("--start", type=int, default=2046)
    parser.add_argument("--stop", type=int, default=2086)
    parser.add_argument(
        "--streams", nargs="+", default=["batting", "baserunning", "pitching-fip", "pitching-babip", "fielding"]
    )
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--out", default="plans/data/attribute-review")
    args = parser.parse_args()
    seasons = range(args.start, args.stop + 1)
    rows = []

    if "batting" in args.streams:
        features = sorted(
            {c for cols in [BATTING_CORE, *BATTING_CANDIDATES.values(), *BATTING_SETS.values()] for c in cols}
            - {"bats_left", "bats_switch"}
            - {f"w_{n}" for n in domains.BATTING_SIDE_PAIRS}
            | {"B"}
        )
        table = data.batting_table(args.save, seasons, features)
        add_helper_columns(table)
        rows += review_stream(
            "batting",
            table,
            BATTING_CORE,
            BATTING_CANDIDATES,
            BATTING_SETS,
            "ridge",
            args.seeds,
            args.folds,
            BATTING_SKILLS,
        )
        rows += review_stream(
            "batting",
            table,
            BATTING_CORE,
            BATTING_CANDIDATES,
            BATTING_SETS,
            "boosting",
            args.seeds,
            args.folds,
            BATTING_SKILLS,
        )
    if "baserunning" in args.streams:
        features = sorted(
            {
                c
                for cols in [BASERUNNING_CORE, *BASERUNNING_CANDIDATES.values(), *BASERUNNING_SETS.values()]
                for c in cols
            }
        )
        table = data.baserunning_table(args.save, seasons, features)
        rows += review_stream(
            "baserunning",
            table,
            BASERUNNING_CORE,
            BASERUNNING_CANDIDATES,
            BASERUNNING_SETS,
            "boosting",
            args.seeds,
            args.folds,
        )
    if "pitching-fip" in args.streams:
        features = sorted({c for cols in [PITCH_CORE, *PITCH_CANDIDATES.values(), *PITCH_SETS.values()] for c in cols})
        table = data.pitching_fip_table(args.save, seasons, features)
        rows += review_stream(
            "pitching-fip",
            table,
            PITCH_CORE,
            PITCH_CANDIDATES,
            PITCH_SETS,
            "boosting",
            args.seeds,
            args.folds,
            PITCH_SKILLS,
        )
    if "fielding" in args.streams:
        for position in domains.FIELD_POSITIONS:
            core = FIELD_CORE[position]
            candidates = field_candidates(position)
            features = sorted({c for cols in [core, *candidates.values()] for c in cols})
            table = data.fielding_table(args.save, seasons, position, features)
            rows += review_stream(
                f"fielding-{position}", table, core, candidates, {}, "boosting", args.seeds, args.folds
            )

    result = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out.with_suffix(".csv"), index=False)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 70)
    text = [result.round(4).to_string(index=False)]

    if "pitching-babip" in args.streams:
        table = data.pitching_babip_table(args.save, seasons, BABIP_CANDIDATES)
        reg = clustered_regression(table, BABIP_CANDIDATES)
        text.append(
            "\npitching BABIP: weighted regression of (BABIP - league) on the ratings, player-clustered t\n"
            + reg.round(5).to_string(index=False)
        )
        only = clustered_regression(table, ["PBABIP vR", "PBABIP vL"])
        text.append("\nPBABIP vR / vL only\n" + only.round(5).to_string(index=False))
    summary = "\n".join(text)
    out.with_suffix(".txt").write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()

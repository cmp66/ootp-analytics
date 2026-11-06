"""Check whether the models miss in the same way in the human leagues: pitchers, hitters, baserunning and fielding.

Usage:
    PYTHONPATH=. python scripts/check_human_bias.py                          # every league of config/update.toml, latest season
    PYTHONPATH=. python scripts/check_human_bias.py --leagues TGS27 PBL27 --models pitching-fip batting
    PYTHONPATH=. python scripts/check_human_bias.py --flag-z 2 --reference-seasons 10

The models learn from a computer league (the `models_from` of each league in config/update.toml). For each model and each human
league this takes the miss of the model for every player (outcome minus prediction), removes the league-wide shift (the
`_rel` columns do that in the viewer), and regresses what is left on the player's ratings, each in standard deviations of the
league (a rating that comes in vs-right and vs-left versions counts as their mean). A coefficient is the runs the model misses
per standard deviation of that rating: per 200 IP for the pitching models, per 600 PA for batting and baserunning, per season
for fielding. `pitching-fip` also gets a term for starters (start share of at least half): the runs per 200 IP they differ from
relievers.

The columns are the leagues, a reference per training league (the same models on its last --reference-seasons seasons: it
should be near zero, though in sample), and the pooled coefficient with its z (the leagues weighted by their precision). A
rating is FLAGGED when all the human leagues agree in sign and the pooled |z| is at least --flag-z (default 2.5): that is the
kind of effect worth a calibration, like `pitching_control_per_sd` (config/saves.toml). One season of a league is a few hundred
players, so a single coefficient is rough; look for the sign agreeing between leagues. Run it again when a league has another
season.
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from model import artifact, domains, registry, scales  # noqa: E402
from util.config import get_save  # noqa: E402
from util.seasons import available_seasons  # noqa: E402
from util.update_config import load_update_config  # noqa: E402

MODELS = {  # name -> (stream, position)
    "pitching-fip": (domains.PITCHING_FIP, None),
    "pitching-babip": (domains.PITCHING_BABIP, None),
    "batting": (domains.BATTING, None),
    "baserunning": (domains.BASERUNNING, None),
    **{f"fielding-{p}": (domains.FIELDING, p) for p in domains.FIELD_POSITIONS},
}
UNITS = {
    "pitching-fip": "runs per 200 IP",
    "pitching-babip": "points of BABIP (x 1000)",
    "batting": "runs per 600 PA",
    "baserunning": "runs per 600 PA",
}


def composite(features: list[str]) -> dict[str, list[str]]:
    """The ratings of a model with the vs-right and vs-left versions of one rating together."""
    groups: dict[str, list[str]] = {}
    for feature in features:
        groups.setdefault(re.sub(r" v[RL]$", "", feature), []).append(feature)
    return groups


def regress(y, X: np.ndarray, weight) -> tuple[np.ndarray, np.ndarray]:
    """Weighted least squares: the coefficients and their standard errors (the first column of X is the constant)."""
    sw = np.sqrt(np.asarray(weight, dtype=float))
    Xw, yw = X * sw[:, None], np.asarray(y, dtype=float) * sw
    beta = np.linalg.lstsq(Xw, yw, rcond=None)[0]
    residual = yw - Xw @ beta
    covariance = (residual @ residual) / max(len(yw) - X.shape[1], 1) * np.linalg.pinv(Xw.T @ Xw)
    return beta, np.sqrt(np.diag(covariance))


def standardize(x: pd.Series, weight) -> pd.Series:
    mean = np.average(x, weights=weight)
    sd = np.sqrt(np.average((x - mean) ** 2, weights=weight))
    return (x - mean) / sd if sd > 0 else x * 0.0


def misses(save: str, models_from: str, seasons: list[int], model: str) -> dict | None:
    """Coefficients of the model's miss on the ratings of one league: {n, sd, coef, se} None when it cannot be scored."""
    stream, position = MODELS[model]
    model_set = scales.model_set(get_save(models_from).ratings, get_save(save).scales)
    loaded = artifact.load_artifact(models_from, model_set, model)
    table = registry.build_table(registry.Spec(model, stream, "plain", loaded.family, position), save, seasons)
    frame = table.frame.reset_index(drop=True)
    if len(frame) < 30:
        return None
    weight = frame["weight"].astype(float).to_numpy()
    y = frame["target"] - loaded.predict(frame[loaded.features])
    y = (y - np.average(y, weights=weight)) * (
        1000.0 if model == "pitching-babip" else 1.0
    )  # the league-wide shift is removed
    columns = {}
    for name, features in composite(loaded.features).items():
        rating = frame[features].apply(pd.to_numeric, errors="coerce").mean(axis=1)
        if rating.nunique() > 2 and rating.notna().mean() > 0.9:
            columns[name] = standardize(rating.fillna(rating.mean()), weight)
    if model == "pitching-fip" and "start_share" in frame:
        columns["starter"] = standardize((frame["start_share"] >= 0.5).astype(float), weight)
    if not columns:
        return None
    X = np.column_stack([np.ones(len(frame)), *columns.values()])
    beta, se = regress(y, X, weight)
    return {
        "n": len(frame),
        "sd": float(np.sqrt(np.average(y**2, weights=weight))),
        "coef": dict(zip(columns, beta[1:])),
        "se": dict(zip(columns, se[1:])),
    }


def pool(coefficients: list[float], errors: list[float]) -> tuple[float, float]:
    """The precision weighted mean of the coefficients of several leagues and its z."""
    weights = 1 / np.asarray(errors, dtype=float) ** 2
    mean = float(np.sum(weights * np.asarray(coefficients)) / weights.sum())
    return mean, mean * float(np.sqrt(weights.sum()))


def table(results: dict[str, dict | None], references: dict[str, dict | None], flag_z: float) -> pd.DataFrame:
    """One row per rating: the coefficient (and z) in each league, in each reference, the pooled value and the flag."""
    ratings = list(dict.fromkeys(r for result in results.values() if result for r in result["coef"]))
    rows = []
    for rating in ratings:
        row: dict = {"rating": rating}
        found = [(name, r["coef"][rating], r["se"][rating]) for name, r in results.items() if r and rating in r["coef"]]
        for name, coef, se in found:
            row[name] = f"{coef:+.2f} ({coef / se:+.1f})"
        for name, reference in references.items():
            has = reference and rating in reference["coef"]
            text = f"{reference['coef'][rating]:+.2f} ({reference['coef'][rating] / reference['se'][rating]:+.1f})"
            row[f"ref {name}"] = text if has else ""
        if len(found) >= 2:
            pooled, z = pool([c for _, c, _ in found], [s for _, _, s in found])
            same = len({np.sign(c) for _, c, _ in found}) == 1
            row.update(
                {"pooled": round(pooled, 2), "z": round(z, 1), "FLAG": "<<<" if same and abs(z) >= flag_z else ""}
            )
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--leagues", nargs="+", help="human leagues (default: those of config/update.toml)")
    parser.add_argument("--models", nargs="+", choices=sorted(MODELS), help="models to check (default: all)")
    parser.add_argument("--season", type=int, help="season of the human leagues (default: the latest of each)")
    parser.add_argument(
        "--reference-seasons", type=int, default=6, help="last seasons of the training league (the reference)"
    )
    parser.add_argument("--flag-z", type=float, default=2.5, help="pooled |z| from which a rating is flagged")
    args = parser.parse_args(argv)
    config = load_update_config()
    leagues = args.leagues or list(config)
    unknown = [name for name in leagues if name not in config]
    if unknown:
        print(f"error: not in config/update.toml: {unknown}", file=sys.stderr)
        return 2
    wanted = args.models or list(MODELS)
    for model in wanted:
        results, references = {}, {}
        for name in leagues:
            models_from = config[name].get("models_from", name)
            try:
                seasons = [args.season] if args.season else [available_seasons(name)[-1]]
                results[name] = misses(name, models_from, seasons, model)
                if models_from not in references:
                    trained = available_seasons(models_from)
                    count = args.reference_seasons
                    last = trained[-count:]
                    references[models_from] = misses(models_from, models_from, last, model)
            except (FileNotFoundError, KeyError, ValueError, IndexError) as error:
                print(f"{model} / {name}: skipped ({error})", file=sys.stderr)
                results[name] = None
        sizes = ", ".join(f"{name} {r['n']} (sd {r['sd']:.1f})" for name, r in results.items() if r)
        print(
            f"\n== {model}: miss in {UNITS.get(model, 'runs per season')} per standard deviation of the rating; z in brackets"
        )
        print(f"   players: {sizes}")
        out = table(results, references, args.flag_z)
        print(out.to_string(index=False) if len(out) else "   nothing to show")
    return 0


if __name__ == "__main__":
    sys.exit(main())

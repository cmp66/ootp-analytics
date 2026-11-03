"""Score the players of a season with trained models and write their predicted values to a csv.

Usage:
    python scripts/predict_values.py Model-Sim                       # latest season, models trained on Model-Sim
    python scripts/predict_values.py TGS --models-from Model-Sim     # TGS ratings, models trained on Model-Sim
    python scripts/predict_values.py TGS --season 2046 --out value.csv

Output (default files/<save>/<season>/output/<save>-<season>-predicted-values.csv), one row per player with ratings:
batting_vs_right / _vs_left / batting (runs per 600 PA), baserunning (runs per 600 PA), pitching_fip_* (runs per 200 IP),
pitching_babip_* (BABIP above the league), pitching_babip_runs and pitching_runs (runs per 200 IP), fielding_<position>
(runs per season if he played there) and the `_plain` fallbacks. The ratings set of the models must match the save's.

Each value column also gets a `<column>_rel` twin: the raw value minus the playing-time weighted mean of that column over
the players of the scored league and season (see model/relative.py). The raw columns are the models' values against the
league they were trained on; the _rel columns are values against the league being scored.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from model import artifact, data, registry, scales  # noqa: E402
from util import paths  # noqa: E402
from util.config import get_save  # noqa: E402
from util.seasons import available_seasons  # noqa: E402

INFO_COLUMNS = ["First Name", "Last Name", "nick_name", "Age", "LPOS", "ORG", "TM", "B", "T"]


def load_artifacts(models_from: str, ratings: str) -> dict:
    folder = artifact.models_dir(models_from, ratings)
    if not folder.exists():
        raise FileNotFoundError(f"{folder} not found; run scripts/train_models.py {models_from}")
    found = {}
    for path in sorted(folder.glob("*.joblib")):
        loaded = artifact.load_artifact(models_from, ratings, path.stem)
        found[loaded.name] = loaded
    if not found:
        raise FileNotFoundError(f"no trained models in {folder}")
    return found


def predict_season(save: str, season: int, artifacts: dict) -> pd.DataFrame:
    """Predicted values with a few identifying columns, one row per player in the season's player-data."""
    players = data.read_output(save, season, "player-data").drop_duplicates("ID").set_index("ID")
    scored, shifts = registry.score_season(save, season, artifacts)
    info = players[[c for c in INFO_COLUMNS if c in players.columns]]
    result = info.join(scored).reset_index()
    result.attrs["league_reference"] = shifts
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("save")
    parser.add_argument("--season", type=int, help="default: the latest season with data")
    parser.add_argument("--models-from", help="save whose models to use (default: the same save)")
    parser.add_argument("--out", help="csv to write")
    args = parser.parse_args(argv)
    try:
        save = get_save(args.save)
        models_from = args.models_from or args.save
        if get_save(models_from).ratings != save.ratings:
            raise ValueError(
                f"{models_from} models are for ratings {get_save(models_from).ratings!r}, {args.save} uses {save.ratings!r}"
            )
        season = args.season if args.season is not None else (available_seasons(args.save) or [None])[-1]
        if season is None:
            raise ValueError(f"no seasons with data for {args.save}")
        artifacts = load_artifacts(models_from, scales.model_set(save.ratings, save.scales))
        result = predict_season(args.save, season, artifacts)
    except (KeyError, ValueError, FileNotFoundError) as error:
        print(f"error: {error.args[0] if isinstance(error, KeyError) else error}", file=sys.stderr)
        return 2
    out = Path(args.out) if args.out else paths.output_file(args.save, season, "predicted-values")
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)
    print(f"{len(result)} players, {len(artifacts)} models ({models_from}) -> {out}")
    shifts = result.attrs.get("league_reference", {})
    print(
        "league reference (subtracted for the _rel columns): " + ", ".join(f"{k} {v:+.2f}" for k, v in shifts.items())
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

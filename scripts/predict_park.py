"""Neutral-park event rates of every player of a season, for the park adjustment of the viewer (model/park.py).

Usage: PYTHONPATH=. python scripts/predict_park.py TGS27                       # latest season, models trained on the save's
ratings set PYTHONPATH=. python scripts/predict_park.py TGS27 --season 2046 --models-from Model-Sim

Writes <save>-<season>-park-rates.csv next to the other outputs: one row per player with all the ratings: ID, the share of his
plate appearances from the left side (`left_share`) and his neutral-park rate per PA of home runs (HR), doubles (DOUBLE),
triples (TRIPLE) and hits (H). The models are the `park` folder of the model set (scripts/train_models.py --streams park); a
league with a rating scale profile uses the models trained for it.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from model import data, park, scales  # noqa: E402
from util import paths  # noqa: E402
from util.config import get_save  # noqa: E402
from util.seasons import available_seasons  # noqa: E402


def switch_left_share(save: str, season: int) -> float:
    """The league's share of plate appearances against right-handed pitchers: what a switch hitter takes from the left side."""
    try:
        right = data.read_output(save, season, "right-hitting")["PA"].sum()
        left = data.read_output(save, season, "left-hitting")["PA"].sum()
    except FileNotFoundError, KeyError:  # an export has no split statistics: the usual share of a league
        return 0.72
    return float(right / (right + left)) if right + left > 0 else 0.72


def predict_season(save: str, season: int, models_from: str | None = None) -> pd.DataFrame:
    config = get_save(save)
    models = park.load_models(models_from or save, scales.model_set(config.ratings, config.scales))
    players = data.read_output(save, season, "player-data").drop_duplicates("ID").set_index("ID")
    ratings = players[park.FEATURES].apply(pd.to_numeric, errors="coerce")
    out = park.rates(ratings, models)
    out.insert(
        0, "left_share", park.left_share(pd.to_numeric(players["B"], errors="coerce"), switch_left_share(save, season))
    )
    return out.dropna(subset=list(park.EVENTS)).reset_index()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("save")
    parser.add_argument("--season", type=int, help="default: the latest season with data")
    parser.add_argument("--models-from", help="save whose models to use (default: the same save)")
    args = parser.parse_args(argv)
    try:
        season = args.season if args.season is not None else (available_seasons(args.save) or [None])[-1]
        if season is None:
            raise ValueError(f"no seasons with data for {args.save}")
        result = predict_season(args.save, season, args.models_from)
    except (KeyError, ValueError, FileNotFoundError) as error:
        print(f"error: {error.args[0] if isinstance(error, KeyError) else error}", file=sys.stderr)
        return 2
    target = paths.output_file(args.save, season, "park-rates")
    result.to_csv(target, index=False)
    print(f"{len(result)} players -> {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

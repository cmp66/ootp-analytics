"""Score the players of a season with the outlook models (plan 16, stage 4).

Usage:
    PYTHONPATH=. python scripts/predict_outlook.py Model-Sim                       # latest season, models trained on Model-Sim
    PYTHONPATH=. python scripts/predict_outlook.py Dugout26 --models-from Model-Sim-26-Std
    PYTHONPATH=. python scripts/predict_outlook.py TGS --season 2046 --models-from Model-Sim --depth-percentile 0.3

Writes two csv files next to the other outputs of the season: <save>-<season>-outlook-slots.csv (one row per player and ranked
slot: value now, rank, tier; for 1, 2, 3 and 5 seasons ahead the median, floor and ceiling of the value, the chances of being
top 10 / top 50 at the slot, `rank_score`, and the value per season with the injury cost) and <save>-<season>-outlook-players.csv
(one row per player: the chance of staying, availability, the injury risk and the depth flag). See model/outlook_scoring.py.
The models and the ratings set must match; all values are relative to the league that is scored.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from model import outlook_scoring  # noqa: E402
from util import paths  # noqa: E402
from util.config import get_save  # noqa: E402
from util.seasons import available_seasons  # noqa: E402


def summary(slots: pd.DataFrame, players: pd.DataFrame, horizons) -> list[str]:
    """A few numbers to eyeball: players, depth share, and how many players the chances add up to at one slot."""
    lines = [
        f"{len(players)} players, {len(slots)} ranked slots; "
        f"depth players (below the cut): {players['depth_player'].mean():.0%}"
    ]
    for slot in ("SS", "SP"):
        part = slots[slots["slot"] == slot]
        if part.empty:
            continue
        h = horizons[-1]
        lines.append(
            f"  {slot}: {len(part)} ranked; sum of top-10 chances {part[f'p_top10_h{h}'].sum():.1f}, "
            f"top-50 {part[f'p_top50_h{h}'].sum():.0f} ({h} seasons ahead)"
        )
    return lines


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("save")
    parser.add_argument("--season", type=int, help="default: the latest season with data")
    parser.add_argument("--models-from", help="save whose models to use (default: the same save)")
    parser.add_argument("--depth-percentile", type=float, default=outlook_scoring.DEPTH_PERCENTILE)
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
        slots, players = outlook_scoring.outlook_for_season(args.save, season, models_from)
        players["depth_player"] = players["best_pct"] < args.depth_percentile
    except (KeyError, ValueError, FileNotFoundError) as error:
        print(f"error: {error.args[0] if isinstance(error, KeyError) else error}", file=sys.stderr)
        return 2
    slots_path = paths.output_file(args.save, season, "outlook-slots")
    players_path = paths.output_file(args.save, season, "outlook-players")
    slots_path.parent.mkdir(parents=True, exist_ok=True)
    slots.round(4).to_csv(slots_path, index=False)
    players.round(4).to_csv(players_path, index=False)
    horizons = sorted(int(c.split("_h")[1]) for c in slots.columns if c.startswith("p_top50_h"))
    print("\n".join(summary(slots, players, horizons)))
    print(f"-> {slots_path}\n-> {players_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

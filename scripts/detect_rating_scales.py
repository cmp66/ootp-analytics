"""Show which scale (20-80 or 1-10) each rating column of a save is on.

Usage:
    PYTHONPATH=. python scripts/detect_rating_scales.py TGS                  # the ratings the value models read, latest season
    PYTHONPATH=. python scripts/detect_rating_scales.py TGS --season 2046 --all   # every numeric column

Reads the processed player-data of the season (scripts/process_seasons.py). Prints, per column, the scale found, the range, the
number of different values and the step between them, so a league that mixes the scales shows which columns are which.
"""

import argparse

import pandas as pd

from model import scales
from util import paths
from util.seasons import available_seasons


def main() -> None:
    parser = argparse.ArgumentParser(description="Show which scale each rating column of a save is on.")
    parser.add_argument("save")
    parser.add_argument("--season", type=int, help="default: the latest season with data")
    parser.add_argument(
        "--all", action="store_true", help="every numeric column, not only the ratings of the value models"
    )
    args = parser.parse_args()

    season = args.season or available_seasons(args.save)[-1]
    frame = pd.read_csv(paths.output_file(args.save, season, "player-data"), low_memory=False)
    columns = [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])] if args.all else None
    table = scales.profile(frame, columns)
    print(f"{args.save} {season}: {len(frame)} players")
    print(table.to_string(index=False))
    counts = table["scale"].value_counts().to_dict()
    print("\n" + ", ".join(f"{n} columns {scale}" for scale, n in counts.items()))


if __name__ == "__main__":
    main()

"""Process the raw OOTP files of a save into the output folders (replaces dataImport.ipynb).

Usage:
    python scripts/process_seasons.py TGS27                       # latest season only
    python scripts/process_seasons.py TGS27 --season 2046         # one season
    python scripts/process_seasons.py Model-Sim --start 2045 --stop 2050
    python scripts/process_seasons.py Model-Sim --start 2080    # 2080 through the latest season
    python scripts/process_seasons.py TGS27 --list                # show the seasons that have data

The save name is a folder under files/ and a table in config/saves.toml, which sets the source (dump or export), the
ratings set and the ballpark file. Output goes to files/<save>/<season>/output/. Run from the repository root.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stats.pipeline import process_season  # noqa: E402
from util import paths  # noqa: E402
from util.config import get_save  # noqa: E402
from util.seasons import available_seasons, select_seasons  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process the raw OOTP files of a save into its output folders.")
    parser.add_argument("save", help="save name: a folder under files/ and a table in config/saves.toml")
    parser.add_argument("--season", type=int, help="process this one season")
    parser.add_argument("--start", type=int, help="first season to process (through the latest unless --stop is given)")
    parser.add_argument("--stop", "--end", type=int, dest="stop", help="last season to process (needs --start)")
    parser.add_argument("--list", action="store_true", help="list the seasons that have data and exit")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    try:
        save = get_save(args.save)
        available = available_seasons(args.save)
        if args.list:
            print(f"{args.save} ({save.source}, ratings {save.ratings}): {available or 'no seasons found'}")
            return 0
        seasons = select_seasons(available, args.start, args.stop, args.season)
    except (KeyError, ValueError) as error:
        print(f"error: {error.args[0] if isinstance(error, KeyError) else error}", file=sys.stderr)
        return 2

    failed = []
    for season in seasons:
        print(f"Processing {args.save} {season} ({save.source}, ratings {save.ratings})...")
        started = time.perf_counter()
        try:
            process_season(args.save, season)
        except Exception as error:  # keep going so one bad season does not stop a long range
            logging.getLogger(__name__).exception("%s %s failed", args.save, season)
            failed.append((season, error))
            continue
        print(f"  wrote {paths.output_dir(args.save, season)} in {time.perf_counter() - started:.1f}s")

    print(f"Done: {len(seasons) - len(failed)} of {len(seasons)} seasons processed.")
    if failed:
        print("Failed seasons: " + ", ".join(str(season) for season, _ in failed), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

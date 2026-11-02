"""Build the longitudinal tables of the outlook models (plan 16, stage 1) for a save.

Usage:
    python scripts/build_outlook_tables.py Model-Sim                      # every season with data
    python scripts/build_outlook_tables.py Model-Sim --start 2046 --stop 2060
    python scripts/build_outlook_tables.py Model-Sim --remeasure-gates    # measure the position minimums again

Writes files/<save>/outlook/players.csv.gz and values.csv.gz (see model/outlook_data.py for the columns) and, the first
time a ratings set is built (or with --remeasure-gates), the position minimums to config/outlook.toml. The values come
from the plan 14 models of the same save (--models-from to use another save's with the same ratings set).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model import outlook_data  # noqa: E402
from util.config import get_save  # noqa: E402
from util.seasons import available_seasons, select_seasons  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("save")
    parser.add_argument("--start", type=int, help="first season (default: the first with data)")
    parser.add_argument("--stop", type=int, help="last season (default: the latest)")
    parser.add_argument("--models-from", help="save whose plan 14 models score the ratings (default: the same save)")
    parser.add_argument("--remeasure-gates", action="store_true", help="measure the position minimums again")
    args = parser.parse_args(argv)
    try:
        save = get_save(args.save)
        models_from = args.models_from or args.save
        if get_save(models_from).ratings != save.ratings:
            raise ValueError(f"{models_from} models are for another ratings set than {args.save}")
        available = available_seasons(args.save)
        seasons = select_seasons(
            available, args.start if args.start is not None else available[0], args.stop or available[-1]
        )
        tables = outlook_data.build_tables(args.save, seasons, models_from, args.remeasure_gates)
        written = outlook_data.write_tables(args.save, tables)
    except (KeyError, ValueError, FileNotFoundError, IndexError) as error:
        print(f"error: {error.args[0] if isinstance(error, KeyError) else error}", file=sys.stderr)
        return 2
    players, values = tables["players"], tables["values"]
    print(f"{len(players)} player-seasons, {len(values)} slot values -> " + ", ".join(str(p) for p in written))
    return 0


if __name__ == "__main__":
    sys.exit(main())

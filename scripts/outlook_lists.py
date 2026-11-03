"""Print or save one of the player lists of config/viewer.toml for a season (plan 16, stage 5).

Usage:
    PYTHONPATH=. python scripts/outlook_lists.py TGS26 --list very_good --slot SS --horizon 3
    PYTHONPATH=. python scripts/outlook_lists.py TGS26 --list trade_targets --org CWS --slot SP --top 20
    PYTHONPATH=. python scripts/outlook_lists.py TGS26 --list draft --out draft.csv
    PYTHONPATH=. python scripts/outlook_lists.py TGS26 --lists                  # the list names

Needs the outlook files of scripts/predict_outlook.py for the season. `--org` is the organization in the save's own notation
(the team abbreviation in an export, the organization number in a dump).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from util.seasons import available_seasons  # noqa: E402
from viewer import queries  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("save")
    parser.add_argument("--list", dest="name", help="a list of config/viewer.toml")
    parser.add_argument("--lists", action="store_true", help="show the list names and exit")
    parser.add_argument("--season", type=int, help="default: the latest season with outlook files")
    parser.add_argument("--org", help="my organization (team abbreviation in an export, number in a dump)")
    parser.add_argument("--horizon", type=int, choices=queries.HORIZONS)
    parser.add_argument("--slot", nargs="+", help="only these slots, for example SS or SP RP")
    parser.add_argument("--top", type=int)
    parser.add_argument("--out", help="write the list to this csv")
    args = parser.parse_args(argv)
    try:
        defaults, specs = queries.load_list_specs()
        if args.lists or not args.name:
            for name, spec in specs.items():
                print(f"{name:18s} {spec.title}")
            return 0
        if args.name not in specs:
            raise ValueError(f"unknown list {args.name!r}; known: {', '.join(specs)}")
        season = args.season if args.season is not None else (available_seasons(args.save) or [None])[-1]
        horizon = args.horizon or defaults.get("horizon", 3)
        frame = queries.load_outlook(args.save, season)
        rows = queries.run_list(
            frame, specs[args.name], horizon, args.org, args.top or defaults.get("top", 50), args.slot
        )
    except (KeyError, ValueError, FileNotFoundError) as error:
        print(f"error: {error.args[0] if isinstance(error, KeyError) else error}", file=sys.stderr)
        return 2
    result = queries.display_table(rows, horizon)
    if args.out:
        result.to_csv(args.out, index=False)
        print(f"{len(result)} rows -> {args.out}")
    else:
        print(f"{specs[args.name].title} ({args.save} {season}, {horizon} seasons ahead)")
        print(result.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

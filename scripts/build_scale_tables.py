"""Build the conversion tables between the 20-80 and 1-10 rating scales from paired dumps (config/rating-scales/<version>.csv).

Usage:
    PYTHONPATH=. python scripts/build_scale_tables.py                       # OOTP26 and OOTP27 of files/Ratings-Analysis
    PYTHONPATH=. python scripts/build_scale_tables.py --root files/Ratings-Analysis --versions OOTP27

Needs, per game version, two dumps of one league taken at the same moment: <root>/<version>/20-80/ and <root>/<version>/1-10/ (the
ratings files players_batting.csv, players_pitching.csv and players_fielding.csv). See model/scales.py.

A league that plays with the game's relative ratings off (PBL27) has another table, `<version>-direct.csv`: one dump of a relative
20-80 run and one of a 1-10 run with relative ratings off, both from the same starting point (the first dumps of Relative80 and
Direct10 of files/Development-Analysis). The two runs hold nearly the same players, so those are paired by player ID and the
others left out:
    PYTHONPATH=. python scripts/build_scale_tables.py --versions OOTP27 --direct \
        files/Development-Analysis/Relative80/21-Oct files/Development-Analysis/Direct10/21-Oct
"""

import argparse
from pathlib import Path

from model import scales


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the 20-80 / 1-10 conversion tables from paired dumps.")
    parser.add_argument("--root", type=Path, default=Path("files/Ratings-Analysis"))
    parser.add_argument("--versions", nargs="+", default=["OOTP26", "OOTP27"])
    parser.add_argument(
        "--direct", nargs=2, type=Path, metavar=("DUMP_20_80", "DUMP_1_10_DIRECT"),
        help="build <version>-direct.csv (relative ratings off) from these two dumps instead of the paired 20-80 / 1-10 tables",
    )  # fmt: skip
    args = parser.parse_args()
    scales.TABLE_DIR.mkdir(parents=True, exist_ok=True)
    for version in args.versions:
        if args.direct:
            counts = scales.paired_counts(*args.direct, partial=True)
            path = scales.TABLE_DIR / f"{scales.table_name(version, relative=False)}.csv"
        else:
            counts = scales.paired_counts(args.root / version / "20-80", args.root / version / "1-10")
            path = scales.TABLE_DIR / f"{version}.csv"
        counts.to_csv(path, index=False)
        print(f"{version}: {counts['column'].nunique()} rating columns, {int(counts['count'].sum())} values -> {path}")


if __name__ == "__main__":
    main()

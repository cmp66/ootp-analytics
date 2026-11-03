"""Build the conversion tables between the 20-80 and 1-10 rating scales from paired dumps (config/rating-scales/<version>.csv).

Usage:
    PYTHONPATH=. python scripts/build_scale_tables.py                       # OOTP26 and OOTP27 of files/Ratings-Analysis
    PYTHONPATH=. python scripts/build_scale_tables.py --root files/Ratings-Analysis --versions OOTP27

Needs, per game version, two dumps of one league taken at the same moment: <root>/<version>/20-80/ and <root>/<version>/1-10/ (the
ratings files players_batting.csv, players_pitching.csv and players_fielding.csv). See model/scales.py.
"""

import argparse
from pathlib import Path

from model import scales


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the 20-80 / 1-10 conversion tables from paired dumps.")
    parser.add_argument("--root", type=Path, default=Path("files/Ratings-Analysis"))
    parser.add_argument("--versions", nargs="+", default=["OOTP26", "OOTP27"])
    args = parser.parse_args()
    scales.TABLE_DIR.mkdir(parents=True, exist_ok=True)
    for version in args.versions:
        counts = scales.paired_counts(args.root / version / "20-80", args.root / version / "1-10")
        path = scales.TABLE_DIR / f"{version}.csv"
        counts.to_csv(path, index=False)
        print(f"{version}: {counts['column'].nunique()} rating columns, {int(counts['count'].sum())} values -> {path}")


if __name__ == "__main__":
    main()

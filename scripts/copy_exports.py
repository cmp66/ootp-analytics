"""Copy the new OOTP exports from the import_export folder of each save into the latest season folder of files/<save>/.

Usage:
    PYTHONPATH=. python scripts/copy_exports.py                 # every save of config/update.toml
    PYTHONPATH=. python scripts/copy_exports.py Dugout26        # one save
    PYTHONPATH=. python scripts/copy_exports.py --dry-run       # say what would be copied

`files/exports/<save>` is a link (symlink) to the import_export folder of the save in the game folder, which differs per machine.
The files wanted are the ones already in `files/<save>/<season>/ratings/` and `.../finance/` (the latest season of the save); any
other file in the export folder is ignored.

  ratings  The files are one export taken together: either all of them are newer than the copy in the season folder or none is.
           All newer: they are copied. None newer: nothing is copied (nothing was exported). A mix, a missing file, or a file
           with a repeated B / T column or without ID: nothing is copied for the save and the problems are listed, because the
           export went wrong.
  finance  Exported now and then: a file that is not newer than the copy (or is not in the export folder) is skipped.

"Newer" means a later modified time. Copies keep the modified time of the export, so a second run finds nothing new.
The exit code is 1 when a save had problems.

Every new ratings export is also kept: the files are written gzipped to `files/<save>/archive/<season>/<in-game date>/`, the date
being the one of the export (found from the dates of birth and ages in it, util/ages.py). A save has one export per sim or so,
and each is a snapshot of every player's ratings at that date: the series shows when ratings change and by how much, which the
single copy in the season folder (replaced by the next export) does not. An export of a date that is already kept is not kept
twice. `--no-archive` skips it, and `--keep-current` keeps the ratings files that are in the season folder now (the last export,
which nothing has archived yet) and copies nothing.
"""

import argparse
import csv
import gzip
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from util import ages, paths  # noqa: E402
from util.seasons import available_seasons  # noqa: E402
from util.update_config import load_update_config  # noqa: E402

EXPORTS = paths.FILES_ROOT / "exports"
UNIQUE_COLUMNS = ("ID", "B", "T")  # the importer reads B and T as L / R / S: a second text version of them breaks it


@dataclass
class Plan:
    save: str
    season: int
    copy: list[tuple[Path, Path]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)


def header(path: Path) -> list[str]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return next(csv.reader(f), [])


def newer(source: Path, target: Path) -> bool:
    return source.stat().st_mtime > target.stat().st_mtime


def header_problems(source: Path) -> list[str]:
    columns = header(source)
    out = [f"{source.name}: no {c} column" for c in UNIQUE_COLUMNS if c not in columns]
    out += [f"{source.name}: {c} appears {columns.count(c)} times" for c in UNIQUE_COLUMNS if columns.count(c) > 1]
    return out


def plan_save(save: str, exports: Path = EXPORTS, season: int | None = None) -> Plan:
    seasons = available_seasons(save)
    season = season or (seasons[-1] if seasons else 0)
    plan = Plan(save, season)
    if not season:
        plan.problems.append(f"no season folder with ratings in files/{save}")
        return plan
    source_dir = exports / save
    if not source_dir.is_dir():
        plan.problems.append(f"{source_dir} does not exist")
        return plan

    targets = sorted(paths.export_ratings_dir(save, season).glob("*.csv"))
    plan.problems += [f"{t.name}: not in the export folder" for t in targets if not (source_dir / t.name).exists()]
    fresh = [
        (source_dir / t.name, t) for t in targets if (source_dir / t.name).exists() and newer(source_dir / t.name, t)
    ]
    if fresh and len(fresh) != len(targets):
        stale = [t.name for t in targets if (source_dir / t.name, t) not in fresh]
        plan.problems.append(f"only some ratings files are new (not new or missing: {', '.join(stale)})")
    elif not fresh and not plan.problems:
        plan.notes.append("ratings: nothing new")
    for source, _ in fresh:
        plan.problems += header_problems(source)
    if not plan.problems:
        plan.copy += fresh

    for target in sorted(paths.finance_dir(save, season).glob("*.csv")):
        source = source_dir / target.name
        if not source.exists():
            plan.notes.append(f"finance: {target.name} is not in the export folder, skipped")
        elif not newer(source, target):
            plan.notes.append("finance: not new, skipped")
        elif plan.problems:
            plan.notes.append("finance: not copied because the ratings files have problems")
        else:
            plan.copy.append((source, target))
    return plan


def export_date_of(sources: list[Path]) -> pd.Timestamp | None:
    """The in-game date of an export: the one every player's date of birth and whole age agree on."""
    frames = [pd.read_csv(path, usecols=["DOB", "Age"], low_memory=False) for path in sources if "DOB" in header(path)]
    if not frames:
        return None
    both = pd.concat(frames)
    return ages.export_date(pd.to_datetime(both["DOB"], format="%m/%d/%Y", errors="coerce"), both["Age"])


def archive_ratings(plan: Plan, dry_run: bool = False) -> str:
    """Keep the new ratings files of a save (gzipped) in the archive folder of their in-game date; returns what was done."""
    ratings = [source for source, target in plan.copy if target.parent.name == "ratings"]
    if not ratings:
        return ""
    date = export_date_of(ratings)
    if date is None:
        return "archive: no date of birth in the export, not kept"
    folder = paths.archive_dir(plan.save, plan.season, f"{date:%Y-%m-%d}")
    if folder.is_dir() and any(folder.glob("*.csv.gz")):
        return f"archive: {date:%Y-%m-%d} is already kept"
    if dry_run:
        return f"archive: would keep {len(ratings)} files as {folder.relative_to(paths.FILES_ROOT)}"
    folder.mkdir(parents=True, exist_ok=True)
    for source in ratings:
        target = folder / (source.name + ".gz")
        with open(source, "rb") as raw, gzip.open(target, "wb") as packed:
            shutil.copyfileobj(raw, packed)
        shutil.copystat(source, target)
    return f"archive: kept {len(ratings)} files as {folder.relative_to(paths.FILES_ROOT)}"


def keep_current(saves: list[str], dry_run: bool = False) -> None:
    """Archive the ratings files that are in the latest season folder of each save now, whatever their age."""
    for save in saves:
        seasons = available_seasons(save)
        if not seasons:
            print(f"{save}: no season folder with ratings")
            continue
        files = sorted(paths.export_ratings_dir(save, seasons[-1]).glob("*.csv"))
        plan = Plan(save, seasons[-1], copy=[(f, f) for f in files])
        print(f"{save} (season {plan.season})\n  {archive_ratings(plan, dry_run)}")


def run(saves: list[str], dry_run: bool = False, exports: Path = EXPORTS, archive: bool = True) -> list[Plan]:
    plans = []
    for save in saves:
        plan = plan_save(save, exports)
        plans.append(plan)
        print(f"{save} (season {plan.season})")
        for note in plan.notes:
            print(f"  {note}")
        for problem in plan.problems:
            print(f"  PROBLEM: {problem}")
        if plan.problems:
            print("  nothing copied for this save: fix the export")
            continue
        if archive:
            note = archive_ratings(plan, dry_run)
            if note:
                print(f"  {note}")
        for source, target in plan.copy:
            print(f"  {'would copy' if dry_run else 'copy'} {target.parent.name}/{target.name}")
            if not dry_run:
                shutil.copy2(source, target)
    return plans


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("saves", nargs="*", help="saves to copy (default: every save of config/update.toml)")
    parser.add_argument("--dry-run", action="store_true", help="say what would be copied, copy nothing")
    parser.add_argument("--no-archive", action="store_true", help="do not keep a dated copy of the ratings export")
    parser.add_argument(
        "--keep-current", action="store_true", help="keep the ratings in the season folder now; copy nothing"
    )
    args = parser.parse_args()
    saves = args.saves or list(load_update_config())
    if args.keep_current:
        keep_current(saves, args.dry_run)
        sys.exit(0)
    plans = run(saves, args.dry_run, archive=not args.no_archive)
    sys.exit(1 if any(p.problems for p in plans) else 0)


if __name__ == "__main__":
    main()

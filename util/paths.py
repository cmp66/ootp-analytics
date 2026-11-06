"""Where the files of a save are: raw season folders, ratings, finance, dumps and the processed output files."""

from pathlib import Path

from util.config import PROJECT_ROOT

FILES_ROOT = PROJECT_ROOT / "files"


def save_dir(save: str) -> Path:
    return FILES_ROOT / save


def export_dir(save: str, season: int) -> Path:
    """Season folder for in-game csv exports (stats files live here)."""
    return save_dir(save) / str(season)


def finance_dir(save: str, season: int) -> Path:
    """The financial overview of the league of an export season (one csv with a row per team)."""
    return export_dir(save, season) / "finance"


def export_ratings_dir(save: str, season: int) -> Path:
    return export_dir(save, season) / "ratings"


def archive_dir(save: str, season: int, date: str) -> Path:
    """Where a kept copy of the ratings exports of one in-game date goes (scripts/copy_exports.py)."""
    return export_dir(save, season).parent / "archive" / str(season) / date


def dump_dir(save: str, ratings: str, season: int) -> Path:
    """Season folder of OOTP database csv dumps."""
    return save_dir(save) / "dumps" / ratings / str(season)


def output_dir(save: str, season: int, create: bool = False) -> Path:
    path = export_dir(save, season) / "output"
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def output_file(save: str, season: int, name: str) -> Path:
    """Processed output file, e.g. output_file("TGS27", 2046, "hitting") -> .../TGS27-2046-hitting.csv"""
    return output_dir(save, season) / f"{save}-{season}-{name}.csv"

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


def dump_dir(save: str, ratings: str, season: int) -> Path:
    """Season folder of OOTP database csv dumps."""
    return save_dir(save) / "dumps" / ratings / str(season)


def output_dir(save: str, season: int, create: bool = False) -> Path:
    path = export_dir(save, season) / "output"
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def output_file(save: str, season: int, name: str) -> Path:
    """Processed output file, e.g. output_file("TGS", 2046, "hitting") -> .../TGS-2046-hitting.csv"""
    return output_dir(save, season) / f"{save}-{season}-{name}.csv"

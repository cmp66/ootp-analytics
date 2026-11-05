"""Which seasons a save has (available_seasons, served_seasons) and which of them a script should process (select_seasons)."""

from util import paths
from util.config import get_save


def available_seasons(save_name: str) -> list[int]:
    """Seasons with raw data for a save, oldest first.

    Dump saves keep one folder per season under dumps/<ratings>/; export saves keep one folder per season with a
    ratings/ folder inside, directly under files/<save>/."""
    save = get_save(save_name)
    if save.source == "dump":
        base = paths.save_dir(save_name) / "dumps" / save.ratings
        found = [p for p in base.iterdir() if p.is_dir() and p.name.isdigit()] if base.is_dir() else []
    else:
        base = paths.save_dir(save_name)
        found = (
            [p for p in base.iterdir() if p.is_dir() and p.name.isdigit() and (p / "ratings").is_dir()]
            if base.is_dir()
            else []
        )
    return sorted(int(p.name) for p in found)


def select_seasons(
    available: list[int], start: int | None = None, stop: int | None = None, season: int | None = None
) -> list[int]:
    """Which seasons to process: the latest by default, one named season, or start..stop (inclusive).

    --start alone runs through the latest season; --stop alone is rejected because it has no start."""
    if not available:
        raise ValueError("no seasons found")
    if season is not None:
        if start is not None or stop is not None:
            raise ValueError("use either a single season or a start / stop range, not both")
        start = stop = season
    elif start is None and stop is None:
        return [available[-1]]
    elif start is None:
        raise ValueError("a stop season needs a start season too")
    elif stop is None:
        if start > available[-1]:
            raise ValueError(f"no seasons from {start} on (latest is {available[-1]})")
        stop = available[-1]
    if start > stop:
        raise ValueError(f"start season {start} is after stop season {stop}")
    chosen = [s for s in available if start <= s <= stop]
    missing = [s for s in range(start, stop + 1) if s not in available]
    if season is not None and missing:
        raise ValueError(f"season {season} not found (available: {available[0]} to {available[-1]})")
    if not chosen:
        raise ValueError(f"no seasons between {start} and {stop} (available: {available[0]} to {available[-1]})")
    if missing:
        raise ValueError(f"seasons without data in {start} to {stop}: {missing}")
    return chosen


def served_seasons(save_name: str) -> list[int]:
    """The seasons the viewer can show: those with raw data (available_seasons) and those whose processed outlook files are
    there, which is all a server that only has the outputs of scripts/sync_files.py (the `viewer` set) holds."""
    found = set(available_seasons(save_name))
    base = paths.save_dir(save_name)
    if base.is_dir():
        found |= {
            int(p.name)
            for p in base.iterdir()
            if p.is_dir() and p.name.isdigit() and paths.output_file(save_name, int(p.name), "outlook-slots").exists()
        }
    return sorted(found)

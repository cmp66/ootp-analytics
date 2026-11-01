"""Time each pipeline stage for a few seasons.

Usage: python scripts/profile_pipeline.py [league:season ...]     (default Model-Sim:2045 TGS:2046)
Writes the same output files as the real pipeline.
"""

import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

import stats  # noqa: E402
from stats import batting, fielding, pitching  # noqa: E402
from util import importCSV  # noqa: E402
from util.config import get_save  # noqa: E402


def stage(label, fn, *args, **kwargs):
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    print(f"  {label:34s} {time.perf_counter() - start:6.2f}s")
    return result


def run(league: str, season: int) -> None:
    print(f"{league} {season}")
    source = get_save(league).source
    start = time.perf_counter()
    r, b, br, bl, p, _, _, f, *_ = stage("import", importCSV.import_files, league, season, write_player_data=False)
    b, lg, _, _ = stage("batting", batting.process_batting_data, b, br, bl, r, league, season, source)
    stage("fielding", fielding.process_fielding_data, f, lg, r, league, season)
    stage("pitching", pitching.process_pitching_data, p, r, league, season, lg)
    stage(
        "league averages + player-data",
        stats.add_league_attribute_averages,
        league,
        season,
        r,
    )
    print(f"  {'TOTAL':34s} {time.perf_counter() - start:6.2f}s")


if __name__ == "__main__":
    targets = sys.argv[1:] or ["Model-Sim:2045", "TGS:2046"]
    for target in targets:
        name, year = target.split(":")
        run(name, int(year))

"""Build the small csv fixtures under tests/data from the full OOTP files under files/.

Usage: python scripts/make_test_data.py

Keeps every major league row of one season (so league totals still balance against the team
tables) plus a few rows from other levels and years (so the import filters are exercised), and
only the players those rows refer to.
"""

import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "files"
TARGET = ROOT / "tests" / "data"

DUMP_SAVE, DUMP_RATINGS, DUMP_SEASON = "Model-Sim", "Standard27", 2045
EXPORT_SAVE, EXPORT_SEASON = "TGS", 2046
EXTRA_ROWS = 200
EXTRA_PLAYERS = 100


def sample(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.sample(n=min(n, len(df)), random_state=0)


def build_dump() -> None:
    src = SOURCE / DUMP_SAVE / "dumps" / DUMP_RATINGS / str(DUMP_SEASON)
    dst = TARGET / DUMP_SAVE / "dumps" / DUMP_RATINGS / str(DUMP_SEASON)
    dst.mkdir(parents=True, exist_ok=True)

    kept_ids: set[int] = set()
    for name in ["batting", "pitching", "fielding"]:
        df = pd.read_csv(src / f"players_career_{name}_stats.csv", low_memory=False)
        major = df[(df["year"] == DUMP_SEASON) & (df["level_id"] == 1)]
        minor = sample(
            df[(df["year"] == DUMP_SEASON) & (df["level_id"] != 1)], EXTRA_ROWS
        )
        earlier = sample(
            df[(df["year"] < DUMP_SEASON) & (df["level_id"] == 1)], EXTRA_ROWS
        )
        kept = pd.concat([major, minor, earlier])
        kept.to_csv(dst / f"players_career_{name}_stats.csv", index=False)
        kept_ids |= set(kept["player_id"])

    for name in ["players", "players_batting", "players_fielding", "players_pitching"]:
        df = pd.read_csv(src / f"{name}.csv", low_memory=False)
        df[df["player_id"].isin(kept_ids)].to_csv(dst / f"{name}.csv", index=False)

    for name in [
        "team_batting_stats",
        "team_pitching_stats",
        "team_fielding_stats_stats",
    ]:
        df = pd.read_csv(src / f"{name}.csv", low_memory=False)
        major = df[df["level_id"] == 1]
        pd.concat([major, sample(df[df["level_id"] != 1], 20)]).to_csv(
            dst / f"{name}.csv", index=False
        )


def build_export() -> None:
    src = SOURCE / EXPORT_SAVE / str(EXPORT_SEASON)
    dst = TARGET / EXPORT_SAVE / str(EXPORT_SEASON)
    (dst / "ratings").mkdir(parents=True, exist_ok=True)
    for file in src.glob("*.csv"):
        shutil.copy(file, dst / file.name)

    stat_ids: set[int] = set()
    for file in src.glob("*.csv"):
        df = pd.read_csv(file)
        if (
            "ID" in df.columns and len(df) > 100
        ):  # player stat files, not the 28-row team files
            stat_ids |= set(df["ID"])

    for file in (src / "ratings").glob("*.csv"):
        df = pd.read_csv(file, low_memory=False)
        keep = df["ID"].isin(stat_ids)
        pd.concat([df[keep], sample(df[~keep], EXTRA_PLAYERS)]).to_csv(
            dst / "ratings" / file.name, index=False
        )


if __name__ == "__main__":
    if TARGET.exists():
        shutil.rmtree(TARGET)
    build_dump()
    build_export()
    size = sum(f.stat().st_size for f in TARGET.rglob("*") if f.is_file())
    print(f"wrote {TARGET} ({size / 1e6:.1f} MB)")

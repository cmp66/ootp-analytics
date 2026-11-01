"""Small synthetic leagues used by the stat tests."""

import numpy as np
import pandas as pd

from stats import woba

BATTING_COLUMNS = [
    "AB",
    "H",
    "SO",
    "PA",
    "G",
    "GS",
    "SINGLE",
    "DOUBLE",
    "TRIPLE",
    "HR",
    "R",
    "RBI",
    "HP",
]
BATTING_COLUMNS += ["SB", "CS", "BB", "IBB", "GIDP", "SH", "SF", "UBR"]


def make_ratings(ids, **extra_columns) -> pd.DataFrame:
    """Ratings frame indexed by ID, like importCSV produces."""
    ratings = pd.DataFrame({"ORG": 1, "BABIP": 50, **extra_columns}, index=pd.Index(ids, name="ID"))
    return ratings


def make_batting(n: int = 40, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pa = rng.integers(50, 650, n)
    bb = rng.binomial(pa, 0.09)
    ibb = rng.binomial(bb, 0.05)
    hp = rng.binomial(pa, 0.01)
    sf = rng.binomial(pa, 0.01)
    sh = rng.binomial(pa, 0.005)
    ab = pa - bb - hp - sf - sh
    so = rng.binomial(ab, 0.2)
    hits = rng.binomial(ab, 0.26)
    hr = rng.binomial(hits, 0.12)
    triple = rng.binomial(hits - hr, 0.03)
    double = rng.binomial(hits - hr - triple, 0.2)
    single = hits - hr - triple - double
    sb = rng.binomial(30, 0.3, n)
    df = pd.DataFrame(
        {
            "ID": np.arange(1, n + 1) * 10,
            "AB": ab,
            "H": hits,
            "SO": so,
            "PA": pa,
            "G": pa // 4,
            "GS": pa // 4,
            "SINGLE": single,
            "DOUBLE": double,
            "TRIPLE": triple,
            "HR": hr,
            "R": rng.binomial(pa, 0.12),
            "RBI": rng.binomial(pa, 0.11),
            "HP": hp,
            "SB": sb,
            "CS": rng.binomial(sb, 0.3),
            "BB": bb,
            "IBB": ibb,
            "GIDP": rng.binomial(ab, 0.02),
            "SH": sh,
            "SF": sf,
            "UBR": rng.normal(0, 2, n),
        }
    )
    return df


def batting_league_totals(df: pd.DataFrame) -> dict:
    return {
        "PA": df["PA"].sum(),
        "AB": df["AB"].sum(),
        "R": df["R"].sum(),
        "H": df["H"].sum(),
        "1B": df["SINGLE"].sum(),
        "2B": df["DOUBLE"].sum(),
        "3B": df["TRIPLE"].sum(),
        "HR": df["HR"].sum(),
        "BB": df["BB"].sum(),
        "IBB": df["IBB"].sum(),
        "HP": df["HP"].sum(),
        "SO": df["SO"].sum(),
        "SF": df["SF"].sum(),
        "SH": df["SH"].sum(),
        "GIDP": df["GIDP"].sum(),
        "SB": df["SB"].sum(),
        "CS": df["CS"].sum(),
        "UBR": df["UBR"].sum(),
    }


def league_table(totals: dict) -> pd.DataFrame:
    """The Stat-indexed Value table that process_batting_data builds."""
    table = pd.DataFrame(list(woba.calc_league_data(totals).items()), columns=["Stat", "Value"])
    return table.set_index("Stat")


def make_pitching(n: int = 40, seed: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    games = rng.integers(10, 60, n)
    gs = np.where(np.arange(n) % 3 == 0, games, np.where(np.arange(n) % 3 == 1, 0, games // 2))
    bf = rng.integers(60, 800, n)
    ab = bf - rng.binomial(bf, 0.09) - rng.binomial(bf, 0.01)
    bb = rng.binomial(bf - ab, 0.9)
    hp = bf - ab - bb
    hits = rng.binomial(ab, 0.26)
    hr = rng.binomial(hits, 0.12)
    triple = rng.binomial(hits - hr, 0.03)
    double = rng.binomial(hits - hr - triple, 0.2)
    single = hits - hr - triple - double
    outs = ab - hits + rng.binomial(ab, 0.01)
    return pd.DataFrame(
        {
            "ID": np.arange(1, n + 1) * 10,
            "IP": outs // 3 + (outs % 3) / 10,
            "K": rng.binomial(ab, 0.2),
            "BF": bf,
            "BB": bb,
            "R": rng.binomial(bf, 0.12),
            "ER": rng.binomial(bf, 0.11),
            "G": games,
            "GS": gs,
            "W": 1,
            "L": 1,
            "S": 0,
            "SF": rng.binomial(bf, 0.01),
            "SH": rng.binomial(bf, 0.005),
            "HR": hr,
            "IBB": rng.binomial(bb, 0.05),
            "DP": rng.binomial(ab, 0.02),
            "HA": hits,
            "AB": ab,
            "SINGLE": single,
            "DOUBLE": double,
            "TRIPLE": triple,
            "HP": hp,
            "SB": rng.binomial(20, 0.3, n),
            "CS": rng.binomial(10, 0.3, n),
        }
    )


def pitching_league_totals(df: pd.DataFrame) -> dict:
    return {
        "PA": df["BF"].sum(),
        "AB": df["AB"].sum(),
        "R": df["R"].sum(),
        "H": df["HA"].sum(),
        "1B": df["SINGLE"].sum(),
        "2B": df["DOUBLE"].sum(),
        "3B": df["TRIPLE"].sum(),
        "HR": df["HR"].sum(),
        "BB": df["BB"].sum(),
        "IBB": df["IBB"].sum(),
        "HP": df["HP"].sum(),
        "SO": df["K"].sum(),
        "SF": df["SF"].sum(),
        "SH": df["SH"].sum(),
        "GIDP": df["DP"].sum(),
        "SB": df["SB"].sum(),
        "CS": df["CS"].sum(),
    }


BIZ_COLUMNS = ["BIZ-R", "BIZ-L", "BIZ-E", "BIZ-U", "BIZ-Z", "BIZ-I"]
FIELDING_RATING_COLUMNS = [
    "IF RNG",
    "IF ARM",
    "IF ERR",
    "TDP",
    "OF RNG",
    "OF ARM",
    "OF ERR",
    "C FRM",
    "C ABI",
    "C ARM",
]
POSITION_COLUMNS = ["P", "C", "1B", "2B", "3B", "SS", "LF", "CF", "RF"]


def make_fielding(per_position: int = 6, seed: int = 3, positions=range(1, 10)) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fielding stats (one row per player and position) and matching ratings."""
    rng = np.random.default_rng(seed)
    rows = []
    for pos in positions:
        for i in range(per_position):
            attempts = rng.integers(20, 400, 5)
            made = [rng.binomial(a, 0.9) for a in attempts]
            row = {
                "ID": pos * 1000 + i,
                "POS": pos,
                "IP": float(rng.integers(100, 1200)),
                "DP": int(rng.integers(0, 120)),
                "ARM": float(rng.normal(0, 3)),
                "FRM": float(rng.normal(0, 3)) if pos == 2 else 0.0,
                "E": int(rng.integers(0, 15)),
                "SBA": int(rng.integers(0, 60)) if pos == 2 else 0,
                "RTO": int(rng.integers(0, 20)) if pos == 2 else 0,
            }
            for name, a in zip(["R", "L", "E", "U", "Z"], attempts):
                row[f"BIZ-{name}"] = int(a)
            for name, m in zip(["Rm", "Lm", "Em", "Um", "Zm"], made):
                row[f"BIZ-{name}"] = int(m)
            row["BIZ-I"] = 3
            row["BIZ-Im"] = 0
            rows.append(row)
    stats = pd.DataFrame(rows)
    ids = stats["ID"].unique()
    ratings = make_ratings(ids)
    for column in FIELDING_RATING_COLUMNS + POSITION_COLUMNS:
        ratings[column] = rng.integers(20, 80, len(ratings))
    return stats, ratings

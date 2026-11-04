"""Stage 3 of plans/11-fielding-ability.md: do the fielding columns isolate the fielder?

Runs the real fielding code (no files written) on every Model-Sim dump season and on TGS27 2046 and reports, for the
old and new make-rate definitions and the volume basis:
  - R^2 of the fielder's own range rating,
  - year-to-year correlation for the same player at the same position (Model-Sim),
  - bias t-values for the staff K rate / pbabip / GB rating (Model-Sim) and park AVG (TGS27).

Needs plans/data/biz-team-seasons.csv (scripts/analyze_biz_babip.py).
Usage: PYTHONPATH=. python scripts/validate_fielding_ability.py
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analyze_biz_fielders import BASE, ols
from stats import fielding
from stats.parks import ballpark_file, load_ballparks
from util.config import get_save
from util.importCSV import load_season

logging.disable(logging.WARNING)
ROOT = Path(__file__).resolve().parent.parent
MIN_IP = 800
EMPTY_LEAGUE = pd.DataFrame({"Value": {}})


def run(league: str, season: int):
    ratings, *_rest = load_season(league, season)
    stats = _rest[6]  # fielding
    totals, _, out, _ = fielding.calculate_player_fielding_stats(stats, EMPTY_LEAGUE, ratings)
    out = out.reset_index()
    pos = out["POS"]
    lg = lambda column: pos.map(totals[column])  # noqa: E731
    out["M1_volume"] = out["PlayMadeAASeason"]
    out["M3_old_make_rate"] = lg("PlayAttemptedPerSeason") * out["PlayPercent"] - lg("PlayPerSeason")
    out["M4_tier_adjusted"] = out["wPMAASeason"]
    r = ratings.reindex(out["ID"])
    out["range"] = np.where(pos <= 6, r["IF RNG"].to_numpy(), r["OF RNG"].to_numpy())
    out["season"] = season
    return out[out["POS"].between(3, 9) & (out["IPClean"] >= MIN_IP)], ratings


METRICS = ["M1_volume", "M3_old_make_rate", "M4_tier_adjusted"]


def persistence(df: pd.DataFrame, metric: str) -> tuple[float, int]:
    x = df[["ID", "POS", "season", "ps", metric]].copy()
    x[metric] = x[metric] - x.groupby("ps")[metric].transform("mean")
    y = x.copy()
    y["season"] += 1
    pairs = x.merge(y, on=["ID", "POS", "season"], suffixes=("", "_prev"))
    return pairs[metric].corr(pairs[metric + "_prev"]), len(pairs)


def model_sim() -> None:
    teams = pd.read_csv(ROOT / "plans" / "data" / "biz-team-seasons.csv")
    teams["k_rate"] = teams["k"] / teams["bf"]
    fielders = pd.read_csv(ROOT / "plans" / "data" / "biz-fielder-seasons.csv")[
        ["season", "player_id", "position", "team_id"]
    ].drop_duplicates(["season", "player_id", "position"], keep=False)
    seasons = sorted(int(p.name) for p in BASE.iterdir() if p.name.isdigit())
    d = pd.concat([run("Model-Sim", s)[0] for s in seasons], ignore_index=True)
    d = d.merge(fielders, left_on=["season", "ID", "POS"], right_on=["season", "player_id", "position"])
    d = d.merge(teams[["season", "team_id", "k_rate", "pbabip", "gb_rating"]], on=["season", "team_id"])
    d["ps"] = d["season"].astype(str) + "-" + d["POS"].astype(str)
    d["key"] = d["team_id"].astype(str)
    rows = []
    for m in METRICS:
        r = ols(d, m, ["range", "k_rate", "pbabip", "gb_rating"], "ps", "key")
        yy, pairs = persistence(d, m)
        rows.append(
            {
                "metric": m,
                "R2": r["r2"].iloc[0],
                "year_to_year": yy,
                "t_range": r.loc["range", "t"],
                "t_k_rate": r.loc["k_rate", "t"],
                "t_pbabip": r.loc["pbabip", "t"],
                "t_gb": r.loc["gb_rating", "t"],
            }
        )
    print(f"Model-Sim: {len(d)} fielder-seasons >= {MIN_IP} IP, {pairs} consecutive-season pairs")
    print(pd.DataFrame(rows).round(3).to_string(index=False))


def tgs() -> None:
    save = get_save("TGS27")
    out, ratings = run("TGS27", 2046)
    team = ratings.reindex(out["ID"])
    major = (team["Lev"] == "MLB").to_numpy()
    tm = team["TM"].where(major).to_numpy()
    parks = load_ballparks(ballpark_file(save, 2046))
    out["park_avg"] = [
        parks["AVG"].get(save.team_names.get(t, t), np.nan) if isinstance(t, str) else np.nan for t in tm
    ]
    out["team"] = tm
    out["ps"] = out["POS"].astype(str)
    out = out[out["team"].notna()]
    rows = []
    for m in METRICS:
        r = ols(out, m, ["range", "park_avg"], "ps", "team")
        rows.append(
            {"metric": m, "R2": r["r2"].iloc[0], "t_range": r.loc["range", "t"], "t_park": r.loc["park_avg", "t"]}
        )
    print(f"\nTGS 2046: {len(out)} fielders >= {MIN_IP} IP (current-team attribution)")
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    model_sim()
    tgs()

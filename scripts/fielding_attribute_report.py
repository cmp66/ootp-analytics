"""Stage 6 of plans/11-fielding-ability.md: how much of fielding value do the fielder's ratings explain?

For each infield and outfield position (every Model-Sim dump season, fielders with >= 800 IP) regress the old make-rate
value (overall make rate) and the new tier-adjusted `wPMAASeason` on the ratings that matter at the position. Prints
R^2 for both and the standardized coefficients of the new value. Nothing is written.

Usage: PYTHONPATH=. python scripts/fielding_attribute_report.py [save]
"""

import logging
import sys

import numpy as np
import pandas as pd

from stats import fielding
from util.config import get_save
from util.importCSV import load_season
from util import paths

logging.disable(logging.WARNING)
MIN_IP = 800
EMPTY_LEAGUE = pd.DataFrame({"Value": {}})
IF_COLUMNS = ["IF RNG", "IF ERR", "IF ARM", "TDP", "POSITION"]
OF_COLUMNS = ["OF RNG", "OF ERR", "OF ARM", "POSITION"]
POSITION_COLUMNS = {1: "P", 2: "C", 3: "1B", 4: "2B", 5: "3B", 6: "SS", 7: "LF", 8: "CF", 9: "RF"}


def season_rows(league: str, season: int) -> pd.DataFrame:
    ratings, *rest = load_season(league, season)
    totals, _, out, _ = fielding.calculate_player_fielding_stats(rest[6], EMPTY_LEAGUE, ratings)
    out = out.reset_index()
    pos = out["POS"]
    out["old"] = pos.map(totals["PlayAttemptedPerSeason"]) * out["PlayPercent"] - pos.map(totals["PlayPerSeason"])
    out["new"] = out["wPMAASeason"]
    r = ratings.reindex(out["ID"])
    for column in ["IF RNG", "IF ERR", "IF ARM", "TDP", "OF RNG", "OF ERR", "OF ARM"]:
        out[column] = r[column].to_numpy()
    out["POSITION"] = [r.iloc[i][POSITION_COLUMNS[p]] for i, p in enumerate(pos)]
    out["season"] = season
    return out[out["POS"].between(3, 9) & (out["IPClean"] >= MIN_IP)]


def fit(df: pd.DataFrame, y: str, xs: list[str]) -> tuple[float, pd.Series]:
    d = df[[y, *xs, "season"]].apply(pd.to_numeric, errors="coerce").dropna()
    for c in [y, *xs]:
        d[c] = d[c] - d.groupby("season")[c].transform("mean")
    X = d[xs].to_numpy(dtype=float)
    X = X / X.std(axis=0)
    Y = d[y].to_numpy(dtype=float)
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    r2 = 1 - ((Y - X @ beta) ** 2).sum() / (Y**2).sum()
    return r2, pd.Series(beta, index=xs)


def main() -> None:
    league = sys.argv[1] if len(sys.argv) > 1 else "Model-Sim"
    save = get_save(league)
    base = paths.save_dir(league) / "dumps" / save.ratings
    seasons = sorted(int(p.name) for p in base.iterdir() if p.name.isdigit())
    d = pd.concat([season_rows(league, s) for s in seasons], ignore_index=True)
    rows, coefficients = [], {}
    for pos in range(3, 10):
        sub = d[d["POS"] == pos]
        xs = IF_COLUMNS if pos <= 6 else OF_COLUMNS
        if pos not in (4, 6):
            xs = [x for x in xs if x != "TDP"]
        r2_old, _ = fit(sub, "old", xs)
        r2_new, beta = fit(sub, "new", xs)
        rows.append({"POS": POSITION_COLUMNS[pos], "n": len(sub), "R2_old": r2_old, "R2_new": r2_new})
        coefficients[POSITION_COLUMNS[pos]] = beta
    print(f"{league}: {len(seasons)} seasons, fielders with >= {MIN_IP} IP; season fixed effects")
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print("\nstandardized coefficients of the tier-adjusted value (plays per season per SD of the rating)")
    print(pd.DataFrame(coefficients).round(2).to_string())


if __name__ == "__main__":
    main()

"""Step 4 of plans/10-biz-babip-effect.md: fielder-level view of the staff effects on BIZ chances.

Usage: python scripts/analyze_biz_fielders.py [team_seasons_csv]
One row per (season, team, player, position) for infielders and outfielders with MIN_IP innings.
Position x season fixed effects, so each fielder is compared with the league at his position.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "files" / "Model-Sim" / "dumps" / "Standard27"
MIN_IP = 800
SEASON_IP = {p: 1200 for p in range(3, 10)}
RUN_VALUE = {3: 0.75, 4: 0.75, 5: 0.75, 6: 0.75, 7: 0.9, 8: 0.9, 9: 0.9}


def fielder_rows(season: int) -> pd.DataFrame:
    folder = BASE / str(season)
    f = pd.read_csv(folder / "players_career_fielding_stats.csv", low_memory=False)
    f = f[(f["year"] == season) & (f["level_id"] == 1) & (f["position"].between(3, 9))].copy()
    f["ip_clean"] = f["ip"] + f["ipf"] / 3
    att = [f"opps_{i}" for i in range(5)]
    mad = [f"opps_made_{i}" for i in range(5)]
    f["att"], f["made"] = f[att].sum(axis=1), f[mad].sum(axis=1)
    g = f.groupby(["team_id", "player_id", "position"], as_index=False)[["ip_clean", "att", "made", "e"]].sum()
    g = g[g["ip_clean"] >= MIN_IP]
    r = pd.read_csv(folder / "players_fielding.csv", low_memory=False)[
        ["player_id", "fielding_ratings_infield_range", "fielding_ratings_outfield_range"]
    ]
    g = g.merge(r, on="player_id", how="left")
    g["range"] = np.where(g["position"] <= 6, g["fielding_ratings_infield_range"], g["fielding_ratings_outfield_range"])
    g["season"] = season
    return g.drop(columns=["fielding_ratings_infield_range", "fielding_ratings_outfield_range"])


def ols(d: pd.DataFrame, y: str, xs: list[str], fe: str, cluster: str) -> pd.DataFrame:
    d = d[[y, *xs, fe, cluster]].dropna().copy()
    for c in [y, *xs]:
        d[c] = d[c] - d.groupby(fe)[c].transform("mean")
    X = d[xs].to_numpy()
    sd = X.std(axis=0)
    X = X / sd
    Y = d[y].to_numpy()
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    res = Y - X @ beta
    bread = np.linalg.inv(X.T @ X)
    meat = sum(np.outer(X[i].T @ res[i], X[i].T @ res[i]) for i in d.groupby(cluster).indices.values())
    se = np.sqrt(np.diag(bread @ meat @ bread))
    return pd.DataFrame({"per_sd": beta, "t": beta / se, "sd_x": sd}, index=xs).assign(
        r2=1 - (res**2).sum() / (Y**2).sum(), y_sd=Y.std(), n=len(d)
    )


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "plans" / "data" / "biz-team-seasons.csv"
    teams = pd.read_csv(path)
    teams["k_rate"] = teams["k"] / teams["bf"]
    staff = teams[["season", "team_id", "pbabip", "gb_rating", "k_rate"]]
    seasons = sorted(int(p.name) for p in BASE.iterdir() if p.name.isdigit())
    d = pd.concat([fielder_rows(s) for s in seasons], ignore_index=True).merge(staff, on=["season", "team_id"])
    d["pos_season"] = d["season"].astype(str) + "-" + d["position"].astype(str)
    d["key"] = d["team_id"].astype(str)
    d["made_season"] = d["made"] / d["ip_clean"] * 1200
    d["att_season"] = d["att"] / d["ip_clean"] * 1200
    d["make_rate"] = d["made"] / d["att"]
    print(f"{len(d)} fielder-seasons with >= {MIN_IP} IP, {d['player_id'].nunique()} players")
    pd.set_option("display.width", 160)

    xs_all = ["range", "pbabip", "k_rate", "gb_rating"]
    for y in ["made_season", "att_season", "make_rate"]:
        print(f"\n=== {y}: all IF+OF, position x season FE, team-clustered ===")
        print(ols(d, y, xs_all, "pos_season", "key").round(4).to_string())

    print("\n=== made_season by position group (per +1 SD, then x run value = runs per 1200 IP) ===")
    rows = []
    for name, positions in [("IF", [3, 4, 5, 6]), ("OF", [7, 8, 9])] + [(str(p), [p]) for p in range(3, 10)]:
        sub = d[d["position"].isin(positions)]
        r = ols(sub, "made_season", xs_all, "pos_season", "key")
        rv = RUN_VALUE[positions[0]]
        row = {"pos": name, "n": r["n"].iloc[0], "y_sd": r["y_sd"].iloc[0], "R2": r["r2"].iloc[0]}
        for x in xs_all:
            row[f"{x}_runs"] = r.loc[x, "per_sd"] * rv
            row[f"{x}_t"] = r.loc[x, "t"]
        rows.append(row)
    print(pd.DataFrame(rows).set_index("pos").round(2).to_string())
    runs_sd = {p: d[d["position"] == p]["made_season"].std() * RUN_VALUE[p] for p in range(3, 10)}
    print("\nSD of made_season in runs (total spread among regulars):", {p: round(v, 1) for p, v in runs_sd.items()})

    # how much does the staff part change a fielder's runs: staff-explained part vs range-explained part
    full = ols(d, "made_season", xs_all, "pos_season", "key")
    only_range = ols(d, "made_season", ["range"], "pos_season", "key")
    print(
        "\nR2 within position-season: range only %.3f, range + staff %.3f"
        % (only_range["r2"].iloc[0], full["r2"].iloc[0])
    )
    d.to_csv(ROOT / "plans" / "data" / "biz-fielder-seasons.csv", index=False)


if __name__ == "__main__":
    main()

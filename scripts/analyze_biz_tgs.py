"""Step 6 of plans/10-biz-babip-effect.md: the BIZ tests on TGS27 2046 with the park AVG factor.

Export saves have no team column in the stats, so a player's team is his current MLB team (ratings TM).
Usage: python scripts/analyze_biz_tgs.py
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from stats.parks import load_ballparks, ballpark_file
from util.config import get_save
from util.importCSV import load_season
from util.innings import ip_to_innings

logging.disable(logging.WARNING)
ROOT = Path(__file__).resolve().parent.parent
TIERS = ["R", "L", "E", "U", "Z"]
RUN_VALUE = {3: 0.75, 4: 0.75, 5: 0.75, 6: 0.75, 7: 0.9, 8: 0.9, 9: 0.9}
MIN_IP = 800


def ols(d: pd.DataFrame, y: str, xs: list[str], fe: str | None, cluster: str) -> pd.DataFrame:
    d = d[[y, *xs, cluster] + ([fe] if fe else [])].dropna().copy()
    for c in [y, *xs]:
        d[c] = d[c] - (d.groupby(fe)[c].transform("mean") if fe else d[c].mean())
    X = d[xs].to_numpy()
    sd = X.std(axis=0)
    X = X / sd
    Y = d[y].to_numpy()
    b, *_ = np.linalg.lstsq(X, Y, rcond=None)
    res = Y - X @ b
    bread = np.linalg.inv(X.T @ X)
    meat = sum(np.outer(X[i].T @ res[i], X[i].T @ res[i]) for i in d.groupby(cluster).indices.values())
    se = np.sqrt(np.diag(bread @ meat @ bread))
    return pd.DataFrame({"per_sd": b, "t": b / se, "sd_x": sd}, index=xs).assign(
        r2=1 - (res @ res) / (Y @ Y), y_sd=Y.std(), n=len(d)
    )


def main() -> None:
    save = get_save("TGS27")
    ratings, _, _, _, pitching, _, _, fielding, *_ = load_season("TGS27", 2046)
    ratings = ratings.copy()
    major = ratings["Lev"] == "MLB"
    team_of = ratings["TM"].where(major)
    ratings["team"] = team_of

    parks = load_ballparks(ballpark_file(save, 2046))  # indexed by team name
    names = save.team_names
    ratings["park_avg"] = team_of.map(
        lambda t: parks["AVG"].get(names.get(t, t), np.nan) if isinstance(t, str) else np.nan
    )

    p = pitching.merge(ratings[["team", "PBABIP"]], left_on="ID", right_index=True)
    p = p[p["team"].notna()].copy()
    p["pb"] = pd.to_numeric(p["PBABIP"], errors="coerce")
    p["ipc"] = ip_to_innings(p["IP"])
    teams = p.groupby("team").apply(
        lambda g: pd.Series(
            {
                "pbabip": (g["pb"] * g["BF"]).sum() / g["BF"].sum(),
                "k_rate": g["K"].sum() / g["BF"].sum(),
                "bf": g["BF"].sum(),
                "bip": (g["AB"] - g["K"] - g["HR"] + g["SF"]).sum(),
            }
        ),
        include_groups=False,
    )
    teams["park_avg"] = ratings.groupby("team")["park_avg"].first()

    f = fielding[fielding["POS"].between(3, 9)].merge(
        ratings[["team", "IF RNG", "OF RNG"]], left_on="ID", right_index=True
    )
    f = f[f["team"].notna()].copy()
    f["ipc"] = ip_to_innings(f["IP"])
    f["att"] = f[[f"BIZ-{k}" for k in TIERS]].sum(axis=1)
    f["made"] = f[[f"BIZ-{k}m" for k in TIERS]].sum(axis=1)
    f["range"] = pd.to_numeric(np.where(f["POS"] <= 6, f["IF RNG"], f["OF RNG"]), errors="coerce")
    byteam = f.groupby("team")[["att", "made", "ipc"]].sum()
    teams = teams.join(byteam)
    teams["att_per_bip"] = teams["att"] / teams["bip"]
    teams["make_rate"] = teams["made"] / teams["att"]
    for k in TIERS:
        teams[f"share_{k}"] = f[f"BIZ-{k}"].groupby(f["team"]).sum() / teams["att"]
        teams[f"rate_{k}"] = f[f"BIZ-{k}m"].groupby(f["team"]).sum() / f[f"BIZ-{k}"].groupby(f["team"]).sum()
    teams["share_I"] = f["BIZ-I"].groupby(f["team"]).sum() / (teams["att"] + f["BIZ-I"].groupby(f["team"]).sum())
    teams["key"] = teams.index
    print(
        f"{len(teams)} teams; park PF AVG range {teams.park_avg.min():.3f}-{teams.park_avg.max():.3f}, "
        f"sd {teams.park_avg.std():.3f}; pbabip sd {teams.pbabip.std():.2f}; k_rate sd {teams.k_rate.std():.4f}"
    )
    pd.set_option("display.width", 180)

    print("\n=== team level (28 teams), per +1 SD ===")
    rows = []
    for y in (
        ["att", "bip", "att_per_bip", "share_I", "make_rate"]
        + [f"share_{k}" for k in TIERS]
        + [f"rate_{k}" for k in TIERS]
    ):
        r = ols(teams, y, ["park_avg", "pbabip", "k_rate"], None, "key")
        row = {"y": y, "y_sd": r["y_sd"].iloc[0]}
        for x in r.index:
            row[x], row[f"t_{x}"] = r.loc[x, "per_sd"], r.loc[x, "t"]
        rows.append(row)
    print(pd.DataFrame(rows).set_index("y").round(4).to_string())

    reg = f[(f["ipc"] >= MIN_IP) & f["range"].notna()].merge(
        teams[["pbabip", "k_rate", "park_avg"]], left_on="team", right_index=True
    )
    reg["made_season"] = reg["made"] / reg["ipc"] * 1200
    reg["att_season"] = reg["att"] / reg["ipc"] * 1200
    reg["make_rate"] = reg["made"] / reg["att"]
    reg["pos"] = reg["POS"].astype(int)
    print(f"\n=== fielder level: {len(reg)} fielders with >= {MIN_IP} IP, position FE, team-clustered ===")
    for y in ["made_season", "att_season", "make_rate"]:
        print(f"-- {y}")
        print(ols(reg, y, ["range", "pbabip", "k_rate", "park_avg"], "pos", "team").round(4).to_string())
    rv = reg["pos"].map(RUN_VALUE)
    reg["made_runs"] = reg["made_season"] * rv
    print("-- runs (made_season x run value)")
    print(ols(reg, "made_runs", ["range", "pbabip", "k_rate", "park_avg"], "pos", "team").round(3).to_string())


if __name__ == "__main__":
    main()

"""Review of the calculated hitting, baserunning, pitching and fielding metrics (plans/12-metric-review.md).

Reads the processed output files of one save (default: files/Model-Sim) for every season and reports
  - league invariants (sums that should be 0, clamps, missing and infinite values),
  - repeatability (same player, consecutive seasons) and how much of each metric the player's ratings explain.
Analysis only; nothing is written.

Usage: python scripts/review_metrics.py [files_root] [save]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)


def load(root: Path, save: str, kind: str) -> pd.DataFrame:
    frames = []
    for folder in sorted((root / save).iterdir()):
        if not folder.name.isdigit():
            continue
        path = folder / "output" / f"{save}-{folder.name}-{kind}.csv"
        if path.exists():
            df = pd.read_csv(path, low_memory=False)
            df["season"] = int(folder.name)
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def yy_corr(df: pd.DataFrame, metric: str, keys=("ID",)) -> tuple[float, int]:
    x = df[[*keys, "season", metric]].replace([np.inf, -np.inf], np.nan).dropna()
    x[metric] = x[metric] - x.groupby("season")[metric].transform("mean")
    y = x.copy()
    y["season"] += 1
    pairs = x.merge(y, on=[*keys, "season"], suffixes=("", "_prev"))
    return pairs[metric].corr(pairs[metric + "_prev"]), len(pairs)


def r2(df: pd.DataFrame, metric: str, xs: list[str]) -> float:
    d = df[[metric, *xs, "season"]].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    for c in [metric, *xs]:
        d[c] = d[c] - d.groupby("season")[c].transform("mean")
    X = d[xs].to_numpy(dtype=float)
    X = X / X.std(axis=0)
    Y = d[metric].to_numpy(dtype=float)
    b, *_ = np.linalg.lstsq(X, Y, rcond=None)
    return 1 - ((Y - X @ b) ** 2).sum() / (Y**2).sum()


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "files"
    save = sys.argv[2] if len(sys.argv) > 2 else "Model-Sim"
    hit = load(root, save, "hitting")
    pit = load(root, save, "pitching")
    fld = load(root, save, "fielding")
    pdata = load(root, save, "player-data")
    print(
        f"{save}: {hit['season'].nunique()} seasons; {len(hit)} batter, {len(pit)} pitcher, {len(fld)} fielder rows\n"
    )

    print("== league invariants per season (max abs over seasons) ==")
    sums = hit.groupby("season")[["wRAA", "wRAAAdj", "UBRAA", "wSB", "BSR", "OFF", "OFFAdj"]].sum()
    print((sums.abs().max()).round(2).to_string())
    pit_sums = pit.groupby(["season", "PBaseline"])["WAA"].sum().abs().max()
    print(f"WAA per season and bucket: {pit_sums:.3f}")
    lg = hit.groupby("season")[["lgwSB", "lgUBR", "lgSB_RATE", "lgSTEAL_RATE", "run_value_sb", "run_value_cs"]].first()
    print(lg.describe().loc[["mean", "min", "max"]].round(4).to_string())
    print(f"seasons with lgwSB == 0 (clamped): {(lg['lgwSB'] == 0).sum()}")
    print(
        f"league SB {hit.groupby('season')['SB'].sum().mean():.0f}, CS {hit.groupby('season')['CS'].sum().mean():.0f}, "
        f"break-even success rate {-lg['run_value_cs'].mean() / (lg['run_value_sb'].mean() - lg['run_value_cs'].mean()):.3f} "
        f"vs league {lg['lgSB_RATE'].mean():.3f}"
    )

    print("\n== missing / infinite values in key columns ==")
    for name, df, cols in [
        ("hitting", hit, ["wOBA", "wRAA", "wRAA600", "BSR", "BSR600", "OFF600", "SBAPERCENT", "UBRAA", "wSB"]),
        ("pitching", pit, ["WAA", "WAA200", "RA9", "wOBAa", "KPercent", "pBABIP", "wSBA", "BBPercent", "HRPercent"]),
        ("fielding", fld, ["runsPAdjSeason", "runsPAdj", "wRunsPSeason", "runsPSeason"]),
    ]:
        stats = {c: (int(df[c].isna().sum()), int(np.isinf(df[c]).sum())) for c in cols}
        print(name, {k: v for k, v in stats.items() if v != (0, 0)} or "none", f"of {len(df)} rows")

    print("\n== hitting: repeatability (PA >= 300) and rating R^2 ==")
    h = hit[hit["PA"] >= 300].merge(pdata, on=["ID", "season"], suffixes=("", "_p"))
    h["GIDP_rate"] = h["GIDP"] / h["AB"]
    h["UBRAA_pa"] = h["UBRAA"] / h["PA"] * 600
    h["wSB_pa"] = h["wSB"] / h["PA"] * 600
    h["SB_att_rate"] = (h["SB"] + h["CS"]) / (h["SINGLE"] + h["BB"] + h["HP"])
    h["SB_succ"] = h["SB"] / (h["SB"] + h["CS"])
    hit_ratings = ["CON", "GAP", "POW", "EYE", "K's", "BABIP"]
    speed_ratings = ["SPE", "STE", "RUN", "SR"]
    rows = []
    for metric, xs in [
        ("wOBA", hit_ratings),
        ("wRAA600", hit_ratings),
        ("OFF600", hit_ratings + speed_ratings),
        ("BSR600", speed_ratings),
        ("UBRAA_pa", speed_ratings),
        ("wSB_pa", speed_ratings),
        ("SB_att_rate", speed_ratings),
        ("GIDP_rate", speed_ratings + ["GBT", "FBT"]),
    ]:
        c, n = yy_corr(h, metric)
        rows.append({"metric": metric, "year_to_year": c, "pairs": n, "R2_ratings": r2(h, metric, xs)})
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print("\nstd of BSR components per 600 PA:", h[["UBRAA_pa", "wSB_pa", "BSR600"]].std().round(2).to_dict())
    print("corr(UBRAA_pa, wSB_pa) = %.2f" % h["UBRAA_pa"].corr(h["wSB_pa"]))
    ratio = h["SB_att_rate"].describe()
    print("steal attempt rate:", ratio[["mean", "50%", "max"]].round(3).to_dict())

    print("\n== pitching: repeatability (BF >= 300... IP >= 80) and rating R^2 ==")
    p = pit[pit["IPClean"] >= 80].merge(pdata, on=["ID", "season"], suffixes=("", "_p"))
    p["FIP_like"] = (13 * p["HR"] + 3 * (p["BB"] + p["HP"] - p["IBB"]) - 2 * p["K"]) / p["IPClean"]
    p["RA9_noise"] = p["RA9"]
    pit_ratings = ["STU", "MOV", "HRR", "PBABIP", "CON.1"]
    rows = []
    for metric in ["WAA200", "RA9", "wOBAa", "FIP_like", "KPercent", "BBPercent", "HRPercent", "pBABIP", "wSBA"]:
        c, n = yy_corr(p, metric)
        rows.append({"metric": metric, "year_to_year": c, "pairs": n, "R2_ratings": r2(p, metric, pit_ratings)})
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    print("\n== fielding components by position (IP >= 600) ==")
    f = fld[fld["IPClean"] >= 600]
    comp = f.groupby("POS")[["runsPAdjSeason", "ARMAASeason", "FRMAASeason", "EAAChanceSeason", "DPAASeason"]].std()
    print(comp.round(2).to_string())
    c = f[f["POS"] == 2]
    print(
        "catchers: corr(runsPAdjSeason, FRMAASeason) = %.2f, SD runsPAdjSeason %.1f vs FRMAASeason %.1f"
        % (c["runsPAdjSeason"].corr(c["FRMAASeason"]), c["runsPAdjSeason"].std(), c["FRMAASeason"].std())
    )
    o = f[f["POS"].isin([7, 8, 9])]
    print("outfield: SD runsPAdjSeason %.1f, SD ARMAASeason %.1f" % (o["runsPAdjSeason"].std(), o["ARMAASeason"].std()))


if __name__ == "__main__":
    main()

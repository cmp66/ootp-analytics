"""Stage 5 of plans/13-ability-targets.md: do the new fielding and pitching values isolate the player?

Reads the processed outputs of one save (regenerate them first; see scripts/review_metrics.py) and reports
  - fielding: repeatability and rating R^2 of runsPAdjSeason versus FldRunsSeason per position,
  - pitching: the same for WAA200 versus FIPRuns200, and the BABIP rating effect on BABIPRuns,
  - sums that must be 0, and the FIP weights each season's league data would give (flags a drift from 13 / 3 / 2).
Usage: PYTHONPATH=. python scripts/validate_ability_targets.py [files_root] [save]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.review_metrics import load, r2, yy_corr
from stats.pitching import FIP_BB, FIP_HR, FIP_K

ROOT = Path(__file__).resolve().parent.parent
MIN_FIELD_IP, MIN_PITCH_IP = 800, 80
POSITION_COLUMNS = {1: "P", 2: "C", 3: "1B", 4: "2B", 5: "3B", 6: "SS", 7: "LF", 8: "CF", 9: "RF"}


def derived_fip_weights(root: Path, save: str) -> pd.DataFrame:
    """FIP weights from each season's league linear weights (run value above a ball in play, times 9)."""
    rows = []
    for folder in sorted((root / save).iterdir()):
        if not folder.name.isdigit():
            continue
        out = folder / "output"
        calcs = out / f"{save}-{folder.name}-woba-calcs.csv"
        totals = out / f"{save}-{folder.name}-batting-totals.csv"
        if not (calcs.exists() and totals.exists()):
            continue
        w = pd.read_csv(calcs).set_index("Stat")["Value"]
        t = pd.read_csv(totals).iloc[0]
        minus = w["RUNS_MINUS"]
        hits = t["1B"] + t["2B"] + t["3B"]
        hit_value = sum(t[b] * (w[f"run_value_{b}"] + minus) for b in ["1B", "2B", "3B"]) / hits
        bip_value = hits / (t["AB"] - t["SO"] - t["HR"] + t["SF"]) * hit_value
        rows.append(
            {
                "season": int(folder.name),
                "HR": (w["run_value_hr"] + minus - bip_value) * 9,
                "BB": (w["run_value_bb"] + minus - bip_value) * 9,
                "K": -bip_value * 9,
            }
        )
    return pd.DataFrame(rows).set_index("season")


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "files"
    save = sys.argv[2] if len(sys.argv) > 2 else "Model-Sim"
    pdata = load(root, save, "player-data")
    fld = load(root, save, "fielding")
    pit = load(root, save, "pitching")

    print("== fielding: year-to-year correlation / rating R^2 (IP >= %d) ==" % MIN_FIELD_IP)
    f = fld[(fld["IPClean"] >= MIN_FIELD_IP) & fld["POS"].between(2, 9)].merge(
        pdata, on=["ID", "season"], suffixes=("", "_p")
    )
    f["POSITION"] = [r[POSITION_COLUMNS[p]] for r, p in zip(f.to_dict("records"), f["POS"])]
    rows = []
    for pos in range(2, 10):
        s = f[f["POS"] == pos]
        xs = (
            ["C ABI", "C ARM", "C FRM"]
            if pos == 2
            else (
                ["IF RNG", "IF ERR", "IF ARM", "POSITION"] if pos <= 6 else ["OF RNG", "OF ERR", "OF ARM", "POSITION"]
            )
        )
        row = {"POS": POSITION_COLUMNS[pos], "n": len(s)}
        for metric in ["runsPAdjSeason", "FldRunsSeason"]:
            row[f"yy_{metric}"] = yy_corr(s, metric)[0]
            row[f"R2_{metric}"] = r2(s, metric, xs)
        rows.append(row)
    table = pd.DataFrame(rows).set_index("POS")
    print(table.round(2).to_string())
    catcher_ok = table.loc["C", "yy_FldRunsSeason"] >= 0.9
    print("catcher repeatability >= 0.9:", "ok" if catcher_ok else "FAIL")

    print("\n== pitching: year-to-year correlation / rating R^2 (IP >= %d) ==" % MIN_PITCH_IP)
    p = pit[pit["IPClean"] >= MIN_PITCH_IP].merge(pdata, on=["ID", "season"], suffixes=("", "_p"))
    ratings = ["STU", "MOV", "HRR", "PBABIP", "CON.1"]
    rows = []
    for metric in ["WAA200", "FIPRuns200", "BABIPRuns200"]:
        c, n = yy_corr(p, metric)
        rows.append({"metric": metric, "year_to_year": c, "pairs": n, "R2_ratings": r2(p, metric, ratings)})
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    b = p.dropna(subset=["PBABIP", "pBABIP"]).copy()
    for c in ["PBABIP", "pBABIP"]:
        b[c] = b[c] - b.groupby("season")[c].transform("mean")
    w = np.sqrt(b["BIP"].to_numpy())
    X = np.column_stack([np.ones(len(b)), b["PBABIP"].to_numpy(float)]) * w[:, None]
    y = b["pBABIP"].to_numpy(float) * w
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    res = y - X @ beta
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * (res @ res) / (len(b) - 2))
    print(
        f"\nBABIP per PBABIP rating point (BIP weighted): {beta[1]:.5f} (t {beta[1] / se[1]:.1f}); earlier estimate -0.0006"
    )

    print("\n== sums that must be 0 (max abs over seasons) ==")
    print(
        "FIPRuns per season and bucket: %.2e ; FIPRunsAdj: %.2e ; BABIPRuns per season: %.2e ; WAA decomposition: %.2e"
        % (
            pit.groupby(["season", "PBaseline"])["FIPRuns"].sum().abs().max(),
            pit.groupby(["season", "PBaseline"])["FIPRunsAdj"].sum().abs().max(),
            pit.groupby("season")["BABIPRuns"].sum().abs().max(),
            (pit["WAA"] * 10 - pit["FIPRuns"] - pit["BABIPRuns"] - pit["OtherRuns"]).abs().max(),
        )
    )

    print("\n== FIP weights from the league data vs %s / %s / %s ==" % (FIP_HR, FIP_BB, FIP_K))
    weights = derived_fip_weights(root, save)
    print(weights.describe().loc[["mean", "min", "max"]].round(2).to_string())
    off = {
        name: (weights[name].sub(const).abs() / abs(const)).max()
        for name, const in [("HR", FIP_HR), ("BB", FIP_BB), ("K", FIP_K)]
    }
    print(
        "largest relative difference:",
        {k: round(v, 3) for k, v in off.items()},
        "-> revisit the weights" if max(off.values()) > 0.1 else "-> within 10%",
    )


if __name__ == "__main__":
    main()

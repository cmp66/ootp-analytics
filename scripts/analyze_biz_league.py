"""Step 5 of plans/10-biz-babip-effect.md: do league-wide BABIP ratings move league-wide BIZ outcomes?

One row per Model-Sim season (42 points). Usage: python scripts/analyze_biz_league.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TIERS = ["R", "L", "E", "U", "Z"]


def main() -> None:
    t = pd.read_csv(ROOT / "plans" / "data" / "biz-team-seasons.csv")
    g = t.groupby("season")
    s = pd.DataFrame(
        {
            "lg_pbabip": g["lg_pbabip"].first(),
            "lg_babip_rating": g["lg_babip_rating"].first(),
            "lg_babip": (g["bip"].sum() * 0 + (g["babip_against"].mean())),
            "att_per_bip": g["att"].sum() / g["bip"].sum(),
            "make_rate": g["made"].sum() / g["att"].sum(),
            "share_I": g["att_I"].sum() / (g["att"].sum() + g["att_I"].sum()),
            "share_if": g["att_if"].sum() / g["att"].sum(),
            "bip_per_bf": g["bip"].sum() / g["bf"].sum(),
            "k_rate": g["k"].sum() / g["bf"].sum(),
        }
    )
    for k in TIERS:
        s[f"share_{k}"] = g[f"att_{k}"].sum() / g["att"].sum()
        s[f"rate_{k}"] = g[f"made_{k}"].sum() / g[f"att_{k}"].sum()
    print(
        s[["lg_pbabip", "lg_babip_rating", "lg_babip", "k_rate"]].describe().loc[["mean", "std", "min", "max"]].round(4)
    )
    print("corr(lg_pbabip, lg_babip_rating) = %.2f" % s["lg_pbabip"].corr(s["lg_babip_rating"]))
    rows = []
    for y in (
        ["lg_babip", "att_per_bip", "make_rate", "share_I", "share_if", "bip_per_bf"]
        + [f"share_{k}" for k in TIERS]
        + [f"rate_{k}" for k in TIERS]
    ):
        for xs in (["lg_pbabip"], ["lg_babip_rating"], ["lg_pbabip", "lg_babip_rating", "k_rate"]):
            X = np.column_stack([np.ones(len(s))] + [(s[x] - s[x].mean()) / s[x].std() for x in xs])
            b, *_ = np.linalg.lstsq(X, s[y].to_numpy(), rcond=None)
            res = s[y].to_numpy() - X @ b
            se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * (res @ res) / (len(s) - X.shape[1]))
            row = {"y": y, "model": "+".join(xs), "y_sd": s[y].std()}
            for i, x in enumerate(xs, 1):
                row[f"b_{x}"], row[f"t_{x}"] = b[i], b[i] / se[i]
            rows.append(row)
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("\nper +1 SD of the season-level rating (n = %d seasons)" % len(s))
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

"""Step 3 of plans/10-biz-babip-effect.md: regress BIZ outcomes on team pitcher pbabip.

Usage: python scripts/analyze_biz_babip_models.py [team_seasons_csv]
Season fixed effects (each column demeaned within its season); X columns are standardized so
coefficients are "per 1 SD". Standard errors are cluster-robust by team within the pooled data
only approximately (plain OLS SE x 1, plus a team-clustered version).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TIERS = ["R", "L", "E", "U", "Z"]
RUN_VALUE = 0.78  # blend of 0.75 (IF) and 0.9 (OF) used by fielding runs


def fe_ols(df: pd.DataFrame, y: str, xs: list[str], cluster: str = "team_id") -> pd.DataFrame:
    d = df[[y, *xs, "season", cluster]].dropna().copy()
    for c in [y, *xs]:
        d[c] = d[c] - d.groupby("season")[c].transform("mean")
    X = d[xs].to_numpy()
    X = X / X.std(axis=0)
    Y = d[y].to_numpy()
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    resid = Y - X @ beta
    n, k = X.shape
    bread = np.linalg.inv(X.T @ X)
    meat = sum(np.outer(X[idx].T @ resid[idx], X[idx].T @ resid[idx]) for idx in d.groupby(cluster).indices.values())
    se = np.sqrt(np.diag(bread @ meat @ bread))  # team-clustered
    r2 = 1 - (resid**2).sum() / (Y**2).sum()
    return pd.DataFrame({"per_sd": beta, "se": se, "t": beta / se, "y_sd": Y.std(), "r2_within": r2}, index=xs)


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "plans" / "data" / "biz-team-seasons.csv"
    t = pd.read_csv(path)

    for tier in TIERS:
        t[f"share_{tier}"] = t[f"att_{tier}"] / t["att"]
        t[f"rate_{tier}"] = t[f"made_{tier}"] / t[f"att_{tier}"]
    t["k_rate"] = t["k"] / t["bf"]
    t["bip_per_bf"] = t["bip"] / t["bf"]
    t["make_rate"] = t["made"] / t["att"]
    t["att_per_bip"] = t["att"] / t["bip"]  # chances charged per ball in play
    t["share_I"] = t["att_I"] / (t["att"] + t["att_I"])
    t["share_of"] = t["att_of"] / t["att"]
    t["share_if"] = t["att_if"] / t["att"]
    # made plays a team would have if each tier were made at the league rate: the "mix" part of made
    lg_rate = {k: t[f"made_{k}"].sum() / t[f"att_{k}"].sum() for k in TIERS}
    t["exp_made_mix"] = sum(t[f"att_{k}"] * lg_rate[k] for k in TIERS)
    t["made_over_mix"] = t["made"] - t["exp_made_mix"]
    print("league make rate by tier:", {k: round(v, 3) for k, v in lg_rate.items()})

    cases = {
        "A: pbabip only": ["pbabip"],
        "B: + own-hitter babip (placebo)": ["pbabip", "babip_rating"],
        "C: + range ratings + gb rating": ["pbabip", "if_range", "of_range", "gb_rating"],
        "D: C + staff K rate": ["pbabip", "if_range", "of_range", "gb_rating", "k_rate"],
    }
    ys = (
        [
            "att",
            "att_per_bip",
            "bip",
            "bip_per_bf",
            "bf",
            "k_rate",
            "share_I",
            "share_if",
            "share_of",
            "make_rate",
            "babip_against",
        ]
        + [f"share_{k}" for k in TIERS]
        + [f"rate_{k}" for k in TIERS]
        + ["made", "exp_made_mix", "made_over_mix"]
    )
    pd.set_option("display.width", 160)
    for label, xs in cases.items():
        print(f"\n=== {label} (season FE, team-clustered SE, effects per +1 SD of x) ===")
        rows = []
        for y in [c for c in ys if c not in xs]:
            r = fe_ols(t, y, xs)
            row = {"y": y, "y_sd": r["y_sd"].iloc[0], "R2": r["r2_within"].iloc[0]}
            for x in xs:
                row[f"{x}"] = r.loc[x, "per_sd"]
                row[f"t_{x}"] = r.loc[x, "t"]
            rows.append(row)
        out = pd.DataFrame(rows).set_index("y")
        print(out.round(4).to_string())

    r = fe_ols(t, "made", ["pbabip", "if_range", "of_range", "gb_rating"])
    r2 = fe_ols(t, "made_over_mix", ["pbabip", "if_range", "of_range", "gb_rating"])
    print("\nRuns per team-season for +1 SD of pbabip (SD of team pbabip: %.2f)" % t["pbabip"].std())
    print("  extra plays made: %.1f  -> %.1f runs" % (r.loc["pbabip", "per_sd"], r.loc["pbabip", "per_sd"] * RUN_VALUE))
    print("  of which tier-mix independent (made - mix expected): %.1f" % r2.loc["pbabip", "per_sd"])
    r3 = fe_ols(t, "exp_made_mix", ["pbabip", "if_range", "of_range", "gb_rating"])
    print(
        "  mix-expected plays: %.1f  -> %.1f runs"
        % (r3.loc["pbabip", "per_sd"], r3.loc["pbabip", "per_sd"] * RUN_VALUE)
    )
    print("  team-level SD of made: %.1f plays" % t.groupby("season")["made"].transform(lambda s: s - s.mean()).std())


if __name__ == "__main__":
    main()

"""Estimate how the derived ratings are made of their components and write docs/derived-ratings.md.

OOTP builds two ratings from others: the batting contact rating CON from BABIP and avoid-strikeouts (AVK, "K's"), and
the pitching movement rating MOV from home-run allowed (HRR, "HRA") and PBABIP. The formulas are not published, so the
weights are fitted on the processed player-data of one or more saves (ordinary least squares on all players with
ratings; every rating is a multiple of 5, which explains most of the residual).

Usage: PYTHONPATH=. python scripts/derived_ratings.py [saves ...] [--out docs/derived-ratings.md]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from util import paths  # noqa: E402
from util.config import get_save  # noqa: E402
from util.seasons import available_seasons  # noqa: E402

# (label, derived column, first component, second component)
BATTING = [
    ("current", "CON", "BABIP", "K's"),
    ("vs right-handed pitching", "CON vR", "BA vR", "K vR"),
    ("vs left-handed pitching", "CON vL", "BA vL", "K vL"),
    ("potential", "CON P", "BA P", "K P"),
]
PITCHING = [
    ("current", "MOV", "HRR", "PBABIP"),
    ("vs right-handed batters", "MOV vR", "HRR vR", "PBABIP vR"),
    ("vs left-handed batters", "MOV vL", "HRR vL", "PBABIP vL"),
    ("potential", "MOV P", "HRR P", "PBABIP P"),
]


def load_ratings(save: str, columns: set[str]) -> pd.DataFrame:
    frames = []
    for season in available_seasons(save):
        path = paths.output_file(save, season, "player-data")
        if path.exists():
            frames.append(pd.read_csv(path, usecols=lambda c: c in columns, low_memory=False))
    return pd.concat(frames, ignore_index=True)


def fit(df: pd.DataFrame, derived: str, first: str, second: str) -> dict:
    sub = df[[derived, first, second]].apply(pd.to_numeric, errors="coerce").dropna()
    sub = sub[(sub > 0).all(axis=1)]  # 0 or "-" means not applicable
    y, a, b = (sub[c].to_numpy(float) for c in (derived, first, second))
    X = np.column_stack([np.ones(len(sub)), a, b])
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    free_r2 = 1 - ((y - X @ coef) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    share = float(np.linalg.lstsq((a - b)[:, None], y - b, rcond=None)[0][0])  # weights that add to 1, no intercept
    resid = y - (share * a + (1 - share) * b)
    return {
        "n": len(sub),
        "first": coef[1],
        "second": coef[2],
        "intercept": coef[0],
        "r2": free_r2,
        "share_first": share,
        "resid_sd": resid.std(),
        "resid_mean": resid.mean(),
    }


def section(title: str, rows: list, saves: dict, first_name: str, second_name: str) -> list[str]:
    lines = [f"### {title}", ""]
    lines.append(
        f"| Version | Save | Players | Weight of {first_name} | Weight of {second_name} | Intercept | R² "
        "| Weights adding to 1 | Residual SD |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---|---:|")
    for label, derived, first, second in rows:
        for save, df in saves.items():
            f = fit(df, derived, first, second)
            lines.append(
                f"| {label} (`{derived}` from `{first}`, `{second}`) | {save} | {f['n']:,} | {f['first']:.2f} | "
                f"{f['second']:.2f} | {f['intercept']:+.1f} | {f['r2']:.3f} | "
                f"{f['share_first']:.0%} / {1 - f['share_first']:.0%} | {f['resid_sd']:.1f} |"
            )
    lines.append("")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("saves", nargs="*", default=["Model-Sim", "TGS27"])
    parser.add_argument("--out", default="docs/derived-ratings.md")
    args = parser.parse_args()
    columns = {c for rows in (BATTING, PITCHING) for _, d, a, b in rows for c in (d, a, b)}
    saves = {}
    for name in args.saves:
        get_save(name)
        saves[name] = load_ratings(name, columns)

    lines = [
        "# Derived ratings: CON and MOV",
        "",
        "Two ratings are not independent skills: OOTP computes them from other ratings. They carry no information beyond",
        "their components, which is why the value models leave them out (`model/domains.py`). The weights below are",
        "**estimated from the data** (ordinary least squares over every player with ratings), not read from OOTP.",
        "Regenerate with `PYTHONPATH=. python scripts/derived_ratings.py`.",
        "",
        "## Short version",
        "",
        "| Derived rating | Formula (approximate) | Check |",
        "|---|---|---|",
        "| Batting **CON** (contact) | **CON ≈ 0.5 × BABIP + 0.5 × AVK** (`K's`) "
        "| R² 0.97; residual SD about 1.5 rating points |",
        "| Pitching **MOV** (movement) | **MOV ≈ 0.8 × HRR + 0.2 × PBABIP** (HRR = home runs allowed, `HRA`) "
        "| R² 0.93; residual SD about 2 rating points |",
        "",
        "- The same relations hold for the vs-right, vs-left and potential versions of each rating (potential leans a little",
        "  more on the second component: MOV ≈ 0.7 to 0.75 HRR + 0.25 to 0.3 PBABIP).",
        "- Every rating is a multiple of 5 (20, 25, ..., 80), so the rounding of the three ratings alone leaves a residual",
        "  of roughly 1.5 to 1.8 points even for an exact formula: CON is as close as that allows, MOV slightly looser.",
        "- The fitted weights differ a little between saves (MOV: 0.84 / 0.18 in Model-Sim, 0.79 / 0.23 in TGS27); 80 / 20 is",
        "  the round number inside that range.",
        "- Practical reading: **contact is half BABIP and half strikeout avoidance; movement is mostly home-run suppression",
        "  with a fifth from BABIP suppression.** A pitcher's value therefore already contains his `PBABIP` twice, once",
        "  directly and once through `MOV`, if both are used.",
        "",
        "## Fitted weights",
        "",
        "Free fit: `derived = intercept + w1 × first + w2 × second`. 'Weights adding to 1' fixes the intercept at 0 and",
        "the weights to sum to 1. Residual SD is in rating points for the constrained fit.",
        "",
    ]
    lines += section("Batting: CON from BABIP and AVK", BATTING, saves, "BABIP", "AVK")
    lines += section("Pitching: MOV from HRR and PBABIP", PITCHING, saves, "HRR", "PBABIP")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Stage 2 of the outlook models (plan 16): how ratings develop, what traits and proneness change, who stays in the league.

Usage:
    PYTHONPATH=. python scripts/review_outlook.py Model-Sim     # one version, writes docs/outlook-diagnostics-Model-Sim.md
    PYTHONPATH=. python scripts/review_outlook.py Model-Sim-26-Std --out docs/x.md --csv-dir plans/data/outlook-review

Needs the tables of scripts/build_outlook_tables.py. For one save (one game version) it measures

1. **Attribute development**: for every rating, the change from one season to the next, by age group, explained by the
   age, the starting level, the room to grow (potential minus current where the game gives a potential), the personality
   tiers (separate effects for a low and a high tier against normal, so a reverse effect shows), proneness (durable and
   fragile against normal) and the playing time of the coming season (missed time). Leadership, loyalty and finances are
   included as a check: they should do nothing. Effects are rating points per season; standard errors are clustered by player.
2. **Staying in the league**: the share still in the league 1, 3 and 5 seasons later by entry route and age, and the effect
   of traits at equal value and age.
3. **Aging of value**: the change of the rating-based value from one season to the next by age, and how well today's value
   predicts the value and the tier h seasons ahead (the baseline the outlook models must beat).
4. **Numbers against categories**: how much of the development effect of work ethic and intelligence a model keeps when
   it only sees the tier (what an export gives) instead of the 1-200 number.
"""

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from model import data, outlook_data  # noqa: E402

# rating -> its potential column in player-data (the game gives a potential for these)
BATTING = {"BABIP": "BA P", "GAP": "GAP P", "POW": "POW P", "EYE": "EYE P", "K's": "K P", "CON": "CON P"}
BATTING_SIDES = [f"{a} {s}" for a in ("BA", "GAP", "POW", "EYE", "K") for s in ("vR", "vL")]
PITCHING = {"STU": "STU P", "MOV": "MOV P", "HRR": "HRR P", "PBABIP": "PBABIP P", "CON.1": "CON.1 P"}
PITCHING_SIDES = [f"{a} {s}" for a in ("STU", "CON.1", "PBABIP", "HRR") for s in ("vR", "vL")]
PITCHER_OTHER = ["STM", "HLD", "PIT", "VELO"]
RUNNING = ["SPE", "STE", "RUN", "SR"]
FIELDING = ["IF RNG", "IF ARM", "TDP", "IF ERR", "OF RNG", "OF ARM", "OF ERR"]
CATCHING = ["C ARM", "C ABI", "C FRM"]
PITCH_TYPES = ["FB", "CH", "CB", "SL", "SI", "SP", "CT", "FO", "CC", "SC", "KC", "KN"]
BODY = ["HT", "WT"]

# (attribute, potential column or None, who it applies to)
ATTRIBUTES = (
    [(a, p, "hitters") for a, p in BATTING.items()]
    + [(a, None, "hitters") for a in BATTING_SIDES + RUNNING + FIELDING + BODY]
    + [(a, None, "catchers") for a in CATCHING]
    + [(a, p, "pitchers") for a, p in PITCHING.items()]
    + [(a, None, "pitchers") for a in PITCHING_SIDES + PITCHER_OTHER]
    + [(a, None, "pitchers_with") for a in PITCH_TYPES]
)

AGE_GROUPS = {"young (16-22)": (16, 22), "prime (23-29)": (23, 29), "older (30+)": (30, 60)}
TRAIT_TIERS = [("WE", "L"), ("WE", "H"), ("INT", "L"), ("INT", "H")]
PLACEBO_TIERS = [("LEA", "L"), ("LEA", "H"), ("LOY", "L"), ("LOY", "H"), ("FIN", "L"), ("FIN", "H")]
PRONE_TIERS = [("Prone3", "Durable"), ("Prone3", "Fragile")]
EFFECTS = ["WE_L", "WE_H", "INT_L", "INT_H", "Prone3_Durable", "Prone3_Fragile", "time_next", "gap"]
MIN_ROWS = 400


# ---------------------------------------------------------------- regression


def cluster_ols(y: np.ndarray, x: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """OLS coefficients, cluster-robust standard errors (clustered by `groups`) and R^2."""
    xtx_inv = np.linalg.pinv(x.T @ x)
    beta = xtx_inv @ (x.T @ y)
    resid = y - x @ beta
    scores = pd.DataFrame(x * resid[:, None]).groupby(groups).sum().to_numpy()
    n, k, g = x.shape[0], x.shape[1], scores.shape[0]
    correction = g / (g - 1) * (n - 1) / max(n - k, 1) if g > 1 else 1.0
    cov = correction * xtx_inv @ (scores.T @ scores) @ xtx_inv
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    total = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((resid**2).sum()) / total if total > 0 else float("nan")
    return beta, se, r2


# ---------------------------------------------------------------- the panel of players and the pairs of seasons


def load_panel(save: str, seasons: list[int], log=print) -> pd.DataFrame:
    """One row per player and season: the outlook players table plus every rating the analysis uses."""
    players = outlook_data.read_tables(save)["players"]
    players = players[players["season"].isin(seasons)]
    wanted = list(
        dict.fromkeys(
            [a for a, _, _ in ATTRIBUTES] + [p for _, p, _ in ATTRIBUTES if p] + ["C"]  # position rating: who can catch
        )
    )
    frames = []
    for season in seasons:
        ratings = data.read_output(save, season, "player-data").drop_duplicates("ID")
        keep = [c for c in wanted if c in ratings.columns]
        part = ratings[["ID", *keep]].copy()
        part[keep] = part[keep].apply(pd.to_numeric, errors="coerce")
        part["season"] = season
        frames.append(part)
    log(f"{save}: ratings of {len(seasons)} seasons read")
    ratings = pd.concat(frames, ignore_index=True)
    overlap = [c for c in ratings.columns if c in players.columns and c not in ("ID", "season")]
    return players.drop(columns=overlap).merge(ratings, on=["ID", "season"], how="inner")


def season_pairs(panel: pd.DataFrame) -> pd.DataFrame:
    """Each player-season joined with the same player's next season (suffix `_next`), consecutive seasons only."""
    next_cols = [c for c in panel.columns if c not in ("ID",)]
    later = panel[["ID", *next_cols]].copy()
    later["season"] = later["season"] - 1
    later = later.rename(columns={c: f"{c}_next" for c in next_cols if c != "season"})
    return panel.merge(later, on=["ID", "season"], how="inner")


def applies_to(frame: pd.DataFrame, who: str, attribute: str) -> pd.Series:
    pitcher = frame["LPOS"].isin(outlook_data.PITCHER_LPOS)
    if who == "hitters":
        return ~pitcher
    if who == "catchers":
        return (~pitcher) & (frame["C"].fillna(0) > 0)
    if who == "pitchers":
        return pitcher
    return pitcher & (frame[attribute] > 0) & (frame[f"{attribute}_next"] > 0)  # the pitch exists in both seasons


# ---------------------------------------------------------------- development effects


def design(frame: pd.DataFrame, attribute: str, potential: str | None, numeric_traits: bool = False):
    """Design matrix (as a DataFrame) for the change of an attribute: age, start level, room to grow, traits, missed time."""
    x = pd.DataFrame(index=frame.index)
    x["const"] = 1.0
    x["age"] = frame["Age"] - frame["Age"].mean()
    x["level"] = frame[attribute] - frame[attribute].mean()
    gap = (frame[potential] - frame[attribute]) / 10.0 if potential else None
    if gap is not None:
        x["gap"] = gap.clip(lower=-3, upper=6)
    if numeric_traits:
        for trait in ("WE", "INT"):
            x[f"{trait}_num"] = (frame[trait] - 100.0) / 50.0
    else:
        for trait, tier in TRAIT_TIERS:
            x[f"{trait}_{tier}"] = (frame[f"{trait}_cat"] == tier).astype(float)
    for trait, tier in PLACEBO_TIERS:
        x[f"{trait}_{tier}"] = (frame[f"{trait}_cat"] == tier).astype(float)
    for trait, tier in PRONE_TIERS:
        x[f"{trait}_{tier}"] = (frame[f"{trait}_cat"] == tier).astype(float)
    played = frame["time_share_next"].notna()
    x["time_next"] = frame["time_share_next"].fillna(0.0)
    x["no_stats_next"] = (~played).astype(float)
    return x


def development_effects(pairs: pd.DataFrame, attribute: str, potential: str | None, who: str) -> list[dict]:
    """Effects on the yearly change of one attribute, one row per age group (coefficients in rating points)."""
    needed = [attribute, f"{attribute}_next"]
    if potential:
        needed.append(potential)
    base = pairs[applies_to(pairs, who, attribute)].dropna(subset=needed)
    rows = []
    for label, (low, high) in AGE_GROUPS.items():
        sample = base[base["Age"].between(low, high)]
        if len(sample) < MIN_ROWS:
            continue
        delta = (sample[f"{attribute}_next"] - sample[attribute]).to_numpy(dtype=float)
        row = {"attribute": attribute, "group": label, "n": len(sample), "mean_change": float(delta.mean())}
        fits = {}
        for name, numeric in (("tiers", False), ("numbers", True)):
            x = design(sample, attribute, potential, numeric_traits=numeric)
            x = x.loc[:, x.std(ddof=0).fillna(0).gt(0) | (x.columns == "const")]
            beta, se, r2 = cluster_ols(delta, x.to_numpy(dtype=float), sample["ID"].to_numpy())
            fits[name] = (dict(zip(x.columns, beta)), dict(zip(x.columns, se)), r2)
        coef, se, r2 = fits["tiers"]
        for effect in [*EFFECTS, *[f"{t}_{tier}" for t, tier in PLACEBO_TIERS], "age", "level"]:
            if effect in coef:
                row[effect] = coef[effect]
                row[f"{effect}_t"] = coef[effect] / se[effect] if se[effect] > 0 else float("nan")
        row["r2_tiers"], row["r2_numbers"] = r2, fits["numbers"][2]
        num_coef, num_se, _ = fits["numbers"][0], fits["numbers"][1], None
        for trait in ("WE", "INT"):
            key = f"{trait}_num"
            if key in num_coef:
                row[f"{trait}_per_50"] = num_coef[key]
                row[f"{trait}_per_50_t"] = num_coef[key] / num_se[key] if num_se[key] > 0 else float("nan")
        rows.append(row)
    return rows


def all_development_effects(pairs: pd.DataFrame, log=print) -> pd.DataFrame:
    rows = []
    for attribute, potential, who in ATTRIBUTES:
        if attribute not in pairs.columns or f"{attribute}_next" not in pairs.columns:
            continue
        if potential and potential not in pairs.columns:
            potential = None
        rows += development_effects(pairs, attribute, potential, who)
    out = pd.DataFrame(rows)
    log(f"development effects for {out['attribute'].nunique()} attributes")
    return out


# ---------------------------------------------------------------- survival


def survival_tables(players: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Share still in the league h seasons later by entry route, and the same for the trait tiers at equal age band."""
    new = players[(~players["left_truncated"].astype(bool)) & (players["entry_season"] == players["season"])]
    by_route = new.groupby("entry_route").agg(
        n=("ID", "size"),
        entry_age=("Age", "median"),
        stay_h1=("stay_h1", "mean"),
        stay_h3=("stay_h3", "mean"),
        stay_h5=("stay_h5", "mean"),
    )
    young = players[(players["Age"] <= 22) & players["stay_h3"].notna()]
    rows = []
    for trait in ("WE", "INT", "LEA", "LOY", "FIN", "Prone3"):
        for tier, group in young.groupby(f"{trait}_cat"):
            rows.append({"trait": trait, "tier": tier, "n": len(group), "stay_h3": group["stay_h3"].mean()})
    return by_route, pd.DataFrame(rows)


def survival_effects(players: pd.DataFrame, values: pd.DataFrame) -> pd.DataFrame:
    """Effect of each trait tier on the chance of being in the league 3 seasons later, at equal age, entry route and value.

    Linear probability model on players aged 16-24 with a known 3-season future; the control is the best value the
    player has at any slot (`best_value`, NaN replaced by the 5th percentile and flagged)."""
    best = values.groupby(["season", "ID"])["value"].max().rename("best_value").reset_index()
    sample = players[(players["Age"].between(16, 24)) & players["stay_h3"].notna()].merge(
        best, on=["season", "ID"], how="left"
    )
    floor = sample["best_value"].quantile(0.05)
    x = pd.DataFrame(index=sample.index)
    x["const"] = 1.0
    x["age"] = sample["Age"] - sample["Age"].mean()
    x["best_value"] = sample["best_value"].fillna(floor)
    x["no_value"] = sample["best_value"].isna().astype(float)
    for route in ("hs_draft", "college_draft", "other"):
        x[f"route_{route}"] = (sample["entry_route"] == route).astype(float)
    for trait, tier in [*TRAIT_TIERS, *PLACEBO_TIERS, *PRONE_TIERS]:
        x[f"{trait}_{tier}"] = (sample[f"{trait}_cat"] == tier).astype(float)
    beta, se, r2 = cluster_ols(
        sample["stay_h3"].to_numpy(dtype=float), x.to_numpy(dtype=float), sample["ID"].to_numpy()
    )
    out = pd.DataFrame({"effect": beta, "se": se}, index=x.columns)
    out["t"] = out["effect"] / out["se"]
    out["n"] = len(sample)
    out["r2"] = r2
    return out.loc[
        [c for c in out.index if c.split("_")[0] in {"WE", "INT", "LEA", "LOY", "FIN", "Prone3"} or c == "best_value"]
    ]


# ---------------------------------------------------------------- aging of value and the baseline


def value_aging(players: pd.DataFrame, values: pd.DataFrame) -> pd.DataFrame:
    """Mean change of the slot value from one season to the next by age, for players at MLB level (DH, SP, RP).

    `p_drop` is the share losing more than half a standard deviation of the slot's value among MLB-level players."""
    rows = []
    for slot in ("DH", "SP", "RP"):
        v = values[values["slot"] == slot][["ID", "season", "value"]]
        later = v.assign(season=v["season"] - 1).rename(columns={"value": "value_next"})
        joined = v.merge(later, on=["ID", "season"]).merge(
            players[["ID", "season", "Age", "level"]], on=["ID", "season"]
        )
        joined = joined[joined["level"] == 1]
        joined["change"] = joined["value_next"] - joined["value"]
        threshold = 0.5 * joined["value"].std()
        for age, group in joined.groupby("Age"):
            if len(group) >= 100:
                rows.append(
                    {
                        "slot": slot,
                        "age": int(age),
                        "n": len(group),
                        "change": group["change"].mean(),
                        "sd": group["change"].std(),
                        "p_drop": float((group["change"] < -threshold).mean()),
                        "drop_threshold": threshold,
                    }
                )
    return pd.DataFrame(rows)


def baseline_prediction(
    values: pd.DataFrame, players: pd.DataFrame, last_season: int, horizons=(1, 2, 3, 5)
) -> pd.DataFrame:
    """How well today's slot value predicts the value and the tier h seasons ahead (players in the pool in both seasons)."""
    rows = []
    ages = players[["ID", "season", "Age"]]
    for slot in ("DH", "SS", "CF", "SP"):
        v = values[values["slot"] == slot].merge(ages, on=["ID", "season"])
        for h in horizons:
            later = v[["ID", "season", "value", "tier"]].assign(season=v["season"] - h)
            both = v.merge(later, on=["ID", "season"], suffixes=("", "_h"))
            for label, (low, high) in {"16-21": (16, 21), "22-27": (22, 27), "28+": (28, 60)}.items():
                part = both[both["Age"].between(low, high)]
                if len(part) < 200:
                    continue
                r = part["value"].corr(part["value_h"])
                top = part[part["tier"] != "rest"]
                rows.append(
                    {"slot": slot, "h": h, "ages": label, "n": len(part), "corr": r, "r2": r * r,
                     "top50_now_still_top50": float((top["tier_h"] != "rest").mean()) if len(top) else np.nan}  # fmt: skip
                )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- report


def fmt_effect(row, effect):
    value, t = row.get(effect), row.get(f"{effect}_t")
    if value is None or pd.isna(value):
        return "-"
    text = f"{value:+.2f} ({t:+.1f})"
    return f"**{text}**" if abs(t) >= 3 else text


def development_report(effects: pd.DataFrame) -> list[str]:
    lines = []
    for group in AGE_GROUPS:
        part = effects[effects["group"] == group]
        if part.empty:
            continue
        lines += [
            f"### {group}",
            "",
            "Effects are rating points per season against a normal tier (t-value in brackets, bold when |t| >= 3); "
            "`gap` is per 10 points of room to grow, `time` per full season played against none (an association: better and "
            "younger players also get more playing time, so it is not purely the effect of playing).",
            "",
            "| Attribute | n | Mean change | WE low | WE high | INT low | INT high | Durable | Fragile | Time (next) | Gap |",
            "|---|---:|---:|---|---|---|---|---|---|---|---|",
        ]
        for _, row in part.iterrows():
            lines.append(
                f"| {row['attribute']} | {int(row['n'])} | {row['mean_change']:+.2f} | "
                + " | ".join(fmt_effect(row, e) for e in EFFECTS)
                + " |"
            )
        lines.append("")
    return lines


def trait_summary(effects: pd.DataFrame) -> list[str]:
    """Share of attributes with |t| >= 3 per effect and group: the real traits should stand out from the placebos."""
    columns = [*EFFECTS[:-2], "LEA_L", "LEA_H", "LOY_L", "LOY_H", "FIN_L", "FIN_H"]
    lines = [
        "| Age group | Attributes | " + " | ".join(columns) + " |",
        "|---|---:|" + "---:|" * len(columns),
    ]
    for group in AGE_GROUPS:
        part = effects[effects["group"] == group]
        if part.empty:
            continue
        cells = []
        for column in columns:
            t = part.get(f"{column}_t")
            cells.append("-" if t is None else f"{(t.abs() >= 3).mean():.0%}")
        lines.append(f"| {group} | {len(part)} | " + " | ".join(cells) + " |")
    return lines


def numbers_vs_tiers(effects: pd.DataFrame) -> list[str]:
    """What a tier keeps of the number: R^2 of the change model with tiers against with the 1-200 numbers."""
    lines = ["| Age group | Mean R² with tiers | Mean R² with numbers | Ratio |", "|---|---:|---:|---:|"]
    for group in AGE_GROUPS:
        part = effects[effects["group"] == group]
        if part.empty:
            continue
        a, b = part["r2_tiers"].mean(), part["r2_numbers"].mean()
        lines.append(f"| {group} | {a:.4f} | {b:.4f} | {a / b:.2f} |")
    return lines


def build_report(save, seasons, effects, by_route, by_trait, surv, aging, baseline, ratings) -> str:
    out = [
        f"# Player development diagnostics: {save} ({ratings})",
        "",
        f"Generated {date.today()} by `scripts/review_outlook.py` from seasons {seasons[0]}-{seasons[-1]} "
        "of the long CPU league. "
        "Each game version has its own report: aging, injury and development settings differ between OOTP 26 and 27.",
        "",
        "## 1 How attributes develop and what traits change",
        "",
        "For every rating the change from one season to the next is regressed on age, the starting level, the room to grow "
        "(potential minus current, where the game gives a potential), the personality tiers (work ethic and intelligence "
        "low or high against normal, leadership, loyalty and finances as placebos that should do nothing), proneness (durable "
        "or fragile against normal) and the playing time of the coming season (missed time slows development in OOTP). "
        "Standard errors are clustered by player. A reverse effect shows as opposite signs for the low and the high tier.",
        "",
        "### Share of attributes where an effect is clearly non-zero (|t| >= 3)",
        "",
        *trait_summary(effects),
        "",
        "### What a category keeps of a number (the export gives tiers, the dump numbers)",
        "",
        *numbers_vs_tiers(effects),
        "",
        *development_report(effects),
        "## 2 Staying in the league",
        "",
        "New entrants (first season seen after the table starts) and the share still in the league later:",
        "",
        "| Entry route | Entrants | Median age | 1 season | 3 seasons | 5 seasons |",
        "|---|---:|---:|---:|---:|---:|",
        *[
            f"| {route} | {int(r['n'])} | {r['entry_age']:.0f} | {r['stay_h1']:.0%} | {r['stay_h3']:.0%} | {r['stay_h5']:.0%} |"
            for route, r in by_route.iterrows()
        ],
        "",
        "Share still in the league 3 seasons later, players aged 22 and younger, by trait tier:",
        "",
        "| Trait | Tier | Players | Still there |",
        "|---|---|---:|---:|",
        *[f"| {r['trait']} | {r['tier']} | {int(r['n'])} | {r['stay_h3']:.1%} |" for _, r in by_trait.iterrows()],
        "",
        "Effect on the chance of being in the league 3 seasons later at equal age, entry route and best value "
        "(linear probability model, players aged 16-24; share points):",
        "",
        "| Term | Effect | t |",
        "|---|---:|---:|",
        *[f"| {name} | {r['effect']:+.4f} | {r['t']:+.1f} |" for name, r in surv.iterrows()],
        "",
        "## 3 Aging of value and the baseline",
        "",
        "Mean change of the rating-based value from one season to the next for players at MLB level, by age "
        "(`p_drop`: share losing more than half a standard deviation of the slot's value among MLB-level players in one season):",
        "",
        "| Slot | Age | Players | Mean change | SD | p_drop |",
        "|---|---:|---:|---:|---:|---:|",
        *[
            f"| {r['slot']} | {r['age']} | {r['n']} | {r['change']:+.2f} | {r['sd']:.2f} | {r['p_drop']:.0%} |"
            for _, r in aging.iterrows()
        ],
        "",
        "How well today's value predicts the value h seasons ahead (players ranked at the slot in both seasons) and "
        "whether a top-50 player stays top-50: the baseline the outlook models must beat.",
        "",
        "| Slot | h | Ages | Players | r | R² | Top 50 now, top 50 later |",
        "|---|---:|---|---:|---:|---:|---:|",
        *[
            f"| {r['slot']} | {r['h']} | {r['ages']} | {r['n']} | {r['corr']:.2f} | {r['r2']:.2f} | "
            f"{r['top50_now_still_top50']:.0%} |"
            for _, r in baseline.iterrows()
        ],
        "",
    ]
    return "\n".join(out)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("save")
    parser.add_argument("--out", help="markdown report (default docs/outlook-diagnostics-<save>.md)")
    parser.add_argument("--csv-dir", default="plans/data/outlook-review", help="folder for the csv tables")
    args = parser.parse_args(argv)
    try:
        tables = outlook_data.read_tables(args.save)
        players, values = tables["players"], tables["values"]
        seasons = sorted(players["season"].unique().tolist())
        ratings = outlook_data.get_save(args.save).ratings
        panel = load_panel(args.save, seasons)
    except (KeyError, ValueError, FileNotFoundError) as error:
        print(f"error: {error.args[0] if isinstance(error, KeyError) else error}", file=sys.stderr)
        return 2
    pairs = season_pairs(panel)
    print(f"{len(pairs)} pairs of consecutive seasons")
    effects = all_development_effects(pairs)
    by_route, by_trait = survival_tables(players)
    surv = survival_effects(players, values)
    aging = value_aging(players, values)
    baseline = baseline_prediction(values, players, seasons[-1])
    csv_dir = Path(args.csv_dir)
    csv_dir.mkdir(parents=True, exist_ok=True)
    effects.to_csv(csv_dir / f"{args.save}-development-effects.csv", index=False)
    aging.to_csv(csv_dir / f"{args.save}-value-aging.csv", index=False)
    baseline.to_csv(csv_dir / f"{args.save}-baseline.csv", index=False)
    report = build_report(args.save, seasons, effects, by_route, by_trait, surv, aging, baseline, ratings)
    out = Path(args.out) if args.out else Path("docs") / f"outlook-diagnostics-{args.save}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

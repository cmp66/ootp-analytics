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
    """OLS coefficients, cluster-robust standard errors (clustered by `groups`) and R^2 (NaN without more rows than terms)."""
    if x.shape[0] <= x.shape[1]:
        return np.full(x.shape[1], np.nan), np.full(x.shape[1], np.nan), float("nan")
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


# ---------------------------------------------------------------- entry routes (draft, college, international)

ROUTES = ["hs_draft", "college_draft", "international", "other"]
PCT_BINS = [0, 50, 70, 85, 95, 100.0001]
PCT_LABELS = ["<50", "50-70", "70-85", "85-95", "95+"]
TARGET_AGES = [22, 24, 26, 28]


def best_by_player_season(values: pd.DataFrame) -> pd.DataFrame:
    """Best slot value of every player-season, whether he is top 10 / top 50 at any slot, and his percentile (0-100) among
    all players with a value that season."""
    flagged = values.assign(is_top50=values["tier"] != "rest", is_top10=values["tier"] == "top10")
    best = flagged.groupby(["season", "ID"]).agg(
        best_value=("value", "max"), top50=("is_top50", "max"), top10=("is_top10", "max")
    )
    best = best.reset_index()
    best["pct"] = best.groupby("season")["best_value"].rank(pct=True) * 100
    return best


def entrants(players: pd.DataFrame, best: pd.DataFrame, horizons=(1, 3, 5)) -> pd.DataFrame:
    """New entrants (first season seen after the table starts) with their value at entry and what became of them.

    `top50_h` / `top10_h` are False when the player is gone or no longer ranked, NaN when season + h is beyond the table.
    """
    new = players[(~players["left_truncated"].astype(bool)) & (players["entry_season"] == players["season"])]
    new = new[
        ["ID", "season", "Age", "entry_route", "WE_cat", "INT_cat", "Prone3_cat", *[f"observed_h{h}" for h in horizons]]
    ]
    frame = new.merge(best, on=["season", "ID"], how="left")
    frame = frame.rename(
        columns={"best_value": "entry_value", "pct": "entry_pct", "top50": "entry_top50", "top10": "entry_top10"}
    )
    for h in horizons:
        later = best.assign(season=best["season"] - h)[["season", "ID", "best_value", "top50", "top10"]]
        later = later.rename(columns={"best_value": f"value_h{h}", "top50": f"top50_h{h}", "top10": f"top10_h{h}"})
        frame = frame.merge(later, on=["season", "ID"], how="left")
        observed = frame[f"observed_h{h}"].astype(bool)
        for name in ("top50", "top10"):
            frame[f"{name}_h{h}"] = frame[f"{name}_h{h}"].fillna(False).astype(float).where(observed)
    return frame


def route_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for route in ROUTES:
        part = frame[frame["entry_route"] == route]
        if len(part) < 50:
            continue
        rows.append(
            {
                "route": route,
                "n": len(part),
                "entry_age": part["Age"].median(),
                "entry_pct": part["entry_pct"].median(),
                "top50_h3": part["top50_h3"].mean(),
                "top50_h5": part["top50_h5"].mean(),
                "top10_h5": part["top10_h5"].mean(),
            }
        )
    return pd.DataFrame(rows)


def route_by_level(frame: pd.DataFrame, min_rows: int = 50) -> pd.DataFrame:
    """Chances by route and by the entry value percentile (what a player must reach, at entry, for a given chance)."""
    frame = frame.assign(level=pd.cut(frame["entry_pct"], PCT_BINS, labels=PCT_LABELS, right=False))
    rows = []
    for route in ROUTES:
        for label in PCT_LABELS:
            part = frame[(frame["entry_route"] == route) & (frame["level"] == label)]
            if len(part) < min_rows:
                continue
            rows.append(
                {
                    "route": route,
                    "level": label,
                    "n": len(part),
                    "top50_h3": part["top50_h3"].mean(),
                    "top50_h5": part["top50_h5"].mean(),
                    "top10_h5": part["top10_h5"].mean(),
                }
            )
    return pd.DataFrame(rows)


VALUE_EDGES = [-np.inf, -30, -20, -10, 0, 10, 20, np.inf]
VALUE_LABELS = ["< -30", "-30 to -20", "-20 to -10", "-10 to 0", "0 to 10", "10 to 20", "20+"]


def route_by_value(frame: pd.DataFrame, min_rows: int = 40) -> pd.DataFrame:
    """Chance of being top 50 five seasons after entry by the entry value itself (best slot value, runs per season,
    league-relative): the same for every league of a version, unlike a percentile of one league's pool."""
    frame = frame.assign(band=pd.cut(frame["entry_value"], VALUE_EDGES, labels=VALUE_LABELS))
    rows = []
    for route in ROUTES:
        for label in VALUE_LABELS:
            part = frame[(frame["entry_route"] == route) & (frame["band"] == label)]
            if len(part) >= min_rows:
                rows.append(
                    {
                        "route": route,
                        "band": label,
                        "n": len(part),
                        "top50_h5": part["top50_h5"].mean(),
                        "top10_h5": part["top10_h5"].mean(),
                    }
                )
    return pd.DataFrame(rows)


def top50_at_age(players: pd.DataFrame, best: pd.DataFrame, new: pd.DataFrame, target_age: int) -> pd.DataFrame:
    """Share of entrants of each route and entry level that are top 50 (at any slot) in the season they turn `target_age`.

    Entrants who are gone by then count as not top 50; entrants whose age-`target_age` season is beyond the table, or
    who were already older, are left out. Also the share still in the league and the mean value of those that are."""
    new = new[new["Age"] < target_age].copy()
    new["target_season"] = new["season"] + (target_age - new["Age"])
    last = players["season"].max()
    new = new[new["target_season"] <= last]
    at_age = players[players["Age"] == target_age][["ID", "season"]].merge(best, on=["ID", "season"], how="left")
    at_age = at_age.rename(columns={"season": "target_season", "best_value": "value_at", "top50": "top50_at"})
    joined = new.merge(at_age[["ID", "target_season", "value_at", "top50_at"]], on=["ID", "target_season"], how="left")
    joined["present"] = joined["value_at"].notna()
    joined["top50_at"] = joined["top50_at"].fillna(False).astype(float)
    joined["level"] = pd.cut(joined["entry_pct"], [0, 70, 90, 100.0001], labels=["<70", "70-90", "90+"], right=False)
    rows = []
    for route in ROUTES:
        for level in ("<70", "70-90", "90+"):
            part = joined[(joined["entry_route"] == route) & (joined["level"] == level)]
            if len(part) < 50:
                continue
            rows.append(
                {
                    "age": target_age, "route": route, "level": level, "n": len(part), "present": part["present"].mean(),
                    "top50": part["top50_at"].mean(), "value_if_present": part.loc[part["present"], "value_at"].mean(),
                }  # fmt: skip
            )
    return pd.DataFrame(rows, columns=["age", "route", "level", "n", "present", "top50", "value_if_present"])


def route_effects(frame: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """Route effects on the chance of being top 50 `horizon` seasons later, at equal entry value and entry age.

    Linear probability model on entrants with a known future: entry value (best slot value), age at entry as dummies,
    route against international, work ethic and intelligence tiers, fragile proneness. Clustered by player."""
    column = f"top50_h{horizon}"
    sample = frame.dropna(subset=["entry_value", column])
    x = pd.DataFrame(index=sample.index)
    x["const"] = 1.0
    x["entry_value"] = sample["entry_value"] / 10.0
    for age in sorted(sample["Age"].unique())[1:]:
        x[f"age_{int(age)}"] = (sample["Age"] == age).astype(float)
    for route in ("hs_draft", "college_draft", "other"):
        x[f"route_{route}"] = (sample["entry_route"] == route).astype(float)
    for trait, tier in [*TRAIT_TIERS, ("Prone3", "Fragile")]:
        x[f"{trait}_{tier}"] = (sample[f"{trait}_cat"] == tier).astype(float)
    x = x.loc[:, x.std(ddof=0).fillna(0).gt(0) | (x.columns == "const")]
    beta, se, r2 = cluster_ols(sample[column].to_numpy(dtype=float), x.to_numpy(dtype=float), sample["ID"].to_numpy())
    out = pd.DataFrame({"effect": beta, "se": se}, index=x.columns)
    out["t"] = out["effect"] / out["se"]
    out["n"] = len(sample)
    out["r2"] = r2
    return out.loc[[c for c in out.index if not c.startswith("age_") and c != "const"]]


def age_matters(frame: pd.DataFrame, horizon: int = 3) -> pd.DataFrame:
    """R^2 of the best value `horizon` seasons later (players still ranked) from the entry value alone, with age added, and with
    the route added: how much knowing the age of a prospect adds to knowing his current rating value."""
    sample = frame.dropna(subset=["entry_value", f"value_h{horizon}"])
    y = sample[f"value_h{horizon}"].to_numpy(dtype=float)
    ages = pd.get_dummies(sample["Age"].astype(int), prefix="age", drop_first=True).astype(float)
    routes = pd.get_dummies(sample["entry_route"], prefix="route", drop_first=True).astype(float)
    value = sample[["entry_value"]].astype(float)
    interaction = ages.mul(sample["entry_value"].to_numpy(dtype=float), axis=0).add_prefix("x_")
    rows = []
    for label, parts in [
        ("entry value only", [value]),
        ("+ age", [value, ages]),
        ("+ age x value", [value, ages, interaction]),
        ("+ route", [value, ages, interaction, routes]),
    ]:
        x = np.column_stack([np.ones(len(sample)), *[p.to_numpy() for p in parts]])
        _, _, r2 = cluster_ols(y, x, sample["ID"].to_numpy())
        rows.append({"model": label, "r2": r2, "n": len(sample)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- missed time and proneness


def missed_time(players: pd.DataFrame, best: pd.DataFrame) -> pd.DataFrame:
    """Playing time of MLB-level players (time_share, 1 = a full season for the level) by proneness tier and age group,
    and the effect of the tiers at equal age and value (regression, share points)."""
    mlb = players[(players["primary_level"] == 1) & players["time_share"].notna()].merge(
        best, on=["season", "ID"], how="left"
    )
    mlb = mlb[mlb["Age"].between(20, 40)].copy()
    mlb["group"] = pd.cut(mlb["Age"], [19, 25, 30, 41], labels=["20-25", "26-30", "31+"])
    rows = []
    for group in ("20-25", "26-30", "31+"):
        part = mlb[mlb["group"] == group]
        base = part[part["Prone3_cat"] == "Normal"]["time_share"].mean()
        for tier in ("Durable", "Normal", "Fragile"):
            tier_rows = part[part["Prone3_cat"] == tier]
            rows.append(
                {
                    "group": group, "tier": tier, "n": len(tier_rows), "time_share": tier_rows["time_share"].mean(),
                    "relative_to_normal": tier_rows["time_share"].mean() / base,
                    "under_70": float((tier_rows["time_share"] < 0.7).mean()),
                }  # fmt: skip
            )
    return pd.DataFrame(rows)


def missed_time_effects(players: pd.DataFrame, best: pd.DataFrame) -> pd.DataFrame:
    """Effect of proneness on the playing time of MLB-level players at equal age and value, with the number as well."""
    mlb = players[(players["primary_level"] == 1) & players["time_share"].notna()].merge(
        best, on=["season", "ID"], how="left"
    )
    mlb = mlb[mlb["Age"].between(20, 40) & mlb["best_value"].notna()]
    x = pd.DataFrame(index=mlb.index)
    x["const"] = 1.0
    x["age"] = mlb["Age"] - mlb["Age"].mean()
    x["age_sq"] = x["age"] ** 2
    x["best_value"] = mlb["best_value"] / 10.0
    x["Durable"] = (mlb["Prone3_cat"] == "Durable").astype(float)
    x["Fragile"] = (mlb["Prone3_cat"] == "Fragile").astype(float)
    out = {}
    for name, columns in {"tiers": ["Durable", "Fragile"], "number (per 50)": ["Prone"]}.items():
        design_ = x[["const", "age", "age_sq", "best_value"]].copy()
        if name == "tiers":
            design_[columns] = x[columns]
        else:
            design_["Prone"] = (mlb["Prone"] - 100.0) / 50.0
        beta, se, _ = cluster_ols(
            mlb["time_share"].to_numpy(dtype=float), design_.to_numpy(dtype=float), mlb["ID"].to_numpy()
        )
        for term, b, e in zip(design_.columns, beta, se):
            if term in columns:
                out[f"{term}" if name == "tiers" else "Prone (per 50 points)"] = {
                    "effect": b,
                    "se": e,
                    "t": b / e,
                    "n": len(mlb),
                }
    return pd.DataFrame(out).T


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


def build_report(save, seasons, effects, by_route, by_trait, surv, aging, baseline, ratings, entry) -> str:
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
    out += entry_report(entry)
    return "\n".join(out)


def md_row(*cells) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def entry_report(entry: dict) -> list[str]:
    def pct(x):
        return "-" if pd.isna(x) else f"{x:.0%}"

    lines = [
        "## 4 Missed time and proneness (for the injury-risk column)",
        "",
        "Playing time of MLB-level players (`time_share`: 1 is a full season for the level; benching and call-ups are in it",
        "too, so only the difference between the tiers says something about injuries). By proneness tier and age group:",
        "",
        "| Age | Tier | Player-seasons | Mean time share | Relative to normal | Under 70 % of a season |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for _, r in entry["missed"].iterrows():
        lines.append(
            md_row(
                r["group"],
                r["tier"],
                int(r["n"]),
                f"{r['time_share']:.2f}",
                f"{r['relative_to_normal']:.2f}",
                pct(r["under_70"]),
            )
        )
    lines += [
        "",
        "Effect of proneness on the time share at equal age and value (regression, clustered by player):",
        "",
        "| Term | Effect on time share | t | Player-seasons |",
        "|---|---:|---:|---:|",
    ]
    for name, r in entry["missed_effects"].iterrows():
        lines.append(md_row(name, f"{r['effect']:+.3f}", f"{r['t']:+.1f}", int(r["n"])))
    lines += [
        "",
        "## 5 Entry routes: high school draft, college draft, international signing",
        "",
        "Entrants are players first seen after the table starts. Value is the best rating-based slot value; the entry level is",
        "its percentile among all players of that season. 'Top 50' means top 50 at any slot, `h` seasons after entry (a player",
        "who is gone counts as not top 50).",
        "",
        "| Route | Entrants | Median age | Median entry level | Top 50 after 3 | Top 50 after 5 | Top 10 after 5 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in entry["summary"].iterrows():
        cells = [r["route"], int(r["n"]), f"{r['entry_age']:.0f}", f"{r['entry_pct']:.0f}"]
        lines.append(md_row(*cells, pct(r["top50_h3"]), pct(r["top50_h5"]), pct(r["top10_h5"])))
    lines += [
        "",
        "### What a prospect must reach at entry",
        "",
        "Chance of becoming a top-50 player (any slot) by entry level (percentile of the rating-based value among all players",
        "that season):",
        "",
        "| Route | Entry level | Entrants | Top 50 after 3 | Top 50 after 5 | Top 10 after 5 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for _, r in entry["by_level"].iterrows():
        lines.append(
            md_row(r["route"], r["level"], int(r["n"]), pct(r["top50_h3"]), pct(r["top50_h5"]), pct(r["top10_h5"]))
        )
    lines += [
        "",
        "Same, by the entry value itself (best slot value in runs per season, league-relative; comparable between leagues of",
        "one version, unlike a percentile of one league's pool):",
        "",
        "| Route | Entry value | Entrants | Top 50 after 5 | Top 10 after 5 |",
        "|---|---|---:|---:|---:|",
    ]
    for _, r in entry["by_value"].iterrows():
        lines.append(md_row(r["route"], r["band"], int(r["n"]), pct(r["top50_h5"]), pct(r["top10_h5"])))
    lines += [
        "",
        "### Is there time to develop? Top 50 at a given age",
        "",
        "Entrants of each route and entry level who are top 50 at the age shown (gone counts as no), and the share still in",
        "the league:",
        "",
        "| Age | Route | Entry level | Entrants | Still in the league | Top 50 | Mean value of those present |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for _, r in entry["at_age"].iterrows():
        cells = [r["age"], r["route"], r["level"], int(r["n"]), pct(r["present"]), pct(r["top50"])]
        lines.append(md_row(*cells, f"{r['value_if_present']:+.1f}"))
    lines += [
        "",
        "### Route effects at equal entry value and age",
        "",
        "Linear probability model for being top 50 five seasons after entry (share points; routes against international; age",
        "at entry and entry value controlled; clustered by player):",
        "",
        "| Term | Effect | t |",
        "|---|---:|---:|",
    ]
    for name, r in entry["route_effects"].iterrows():
        lines.append(md_row(name, f"{r['effect']:+.4f}", f"{r['t']:+.1f}"))
    lines += [
        "",
        "### Does the age of a prospect matter?",
        "",
        "Explained variance (R²) of the best value three seasons later, for prospects still ranked:",
        "",
        "| Model | R² | Players |",
        "|---|---:|---:|",
    ]
    for _, r in entry["age_matters"].iterrows():
        lines.append(md_row(r["model"], f"{r['r2']:.3f}", int(r["n"])))
    return lines + [""]


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
    best = best_by_player_season(values)
    new = entrants(players, best)
    entry = {
        "missed": missed_time(players, best),
        "missed_effects": missed_time_effects(players, best),
        "summary": route_summary(new),
        "by_level": route_by_level(new),
        "by_value": route_by_value(new),
        "at_age": pd.concat([top50_at_age(players, best, new, a) for a in TARGET_AGES], ignore_index=True),
        "route_effects": route_effects(new),
        "age_matters": age_matters(new),
    }
    for name in ("summary", "by_level", "by_value", "at_age", "missed"):
        entry[name].to_csv(csv_dir / f"{args.save}-entry-{name}.csv", index=False)
    report = build_report(args.save, seasons, effects, by_route, by_trait, surv, aging, baseline, ratings, entry)
    out = Path(args.out) if args.out else Path("docs") / f"outlook-diagnostics-{args.save}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

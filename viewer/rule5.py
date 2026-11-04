"""Data of the Rule 5 page: pure functions over the outlook frame, no UI.

Your players who can be drafted (the R5 flag of the export), which of them to protect on the 40-man roster, who could be taken
off it to make room, and the players of other organizations you could draft. See config/rule5.toml for the rules.
"""

import tomllib
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from stats.finances import parse_money
from viewer import acquire, depth, minors

RULE5_CONFIG = Path(__file__).resolve().parent.parent / "config" / "rule5.toml"
YES = {"yes", "y", "1", "true"}
PROTECT, LOW_RISK, EXPOSE = "Protect", "Low risk", "Expose"
ORDER = {PROTECT: 0, LOW_RISK: 1, EXPOSE: 2}


@dataclass(frozen=True)
class Settings:
    size: int = 40
    column: str = "ON40"
    ready_percentile: float = 0.20
    potential_percentile: float = 0.90


def load_settings(path: Path = RULE5_CONFIG) -> Settings:
    raw = tomllib.loads(path.read_text()) if path.exists() else {}
    values = {**raw.get("roster", {}), **raw.get("rules", {})}
    unknown = set(values) - set(Settings.__dataclass_fields__)
    if unknown:
        raise ValueError(f"{path}: unknown settings {sorted(unknown)}")
    return replace(Settings(), **values)


def flagged(column: pd.Series) -> pd.Series:
    """True where an export flag says Yes (Yes, Y, 1, True); the rest ('-', 0, No) is False."""
    return column.astype(str).str.strip().str.lower().isin(YES)


def on_forty(frame: pd.DataFrame, settings: Settings) -> tuple[pd.Series, bool]:
    """(per row: on the 40-man roster, whether it is exact). Without the export's flag it is an estimate: the MLB-level players
    and the minor leaguers with time on the secondary roster who cannot be drafted (a draft-eligible player is not on it).
    """
    if settings.column in frame.columns:
        return flagged(frame[settings.column]), True
    years = pd.to_numeric(frame["SECY"], errors="coerce") if "SECY" in frame else pd.Series(0, index=frame.index)
    secondary = years.fillna(0) > 0
    eligible = flagged(frame["R5"]) if "R5" in frame else pd.Series(False, index=frame.index)
    return (frame["level_name"] == "MLB") | (secondary & ~eligible & (frame["level_name"] != "MLB")), False


def player_table(
    frame: pd.DataFrame, org: str, settings: Settings, rules: minors.Rules, version: str = "OOTP27"
) -> pd.DataFrame:
    """The players of `org`, one row each at the position where he stands highest among the MLB players (his readiness for the
    majors): ID, Player, Pos, Age, Level, pitcher, Value (value now there), MLB (his percentile among the MLB players at the
    position), Gain (what he would add over the weakest MLB starter of the position, 0 when nothing), Ceil, Floor, Top50,
    ceiling_pct, Tier, Window, focus, R5 (can be drafted), On40, Owed (his contract) and Years."""
    members = frame[frame["org"] == str(org)]
    rows = members[members["slot"].isin(depth.CHART_SLOTS)].dropna(subset=["value_now"]).copy()
    mlb = frame[
        (frame["level_name"] == "MLB") & ~frame["org"].isin(["FA", "nan", ""]) & frame["slot"].isin(depth.CHART_SLOTS)
    ]
    rows["MLB"] = np.nan
    for slot, group in rows.groupby("slot"):
        peers = mlb.loc[mlb["slot"] == slot, "value_now"].to_numpy(dtype=float)
        rows.loc[group.index, "MLB"] = minors.percentile(group["value_now"], peers)
    best = rows.sort_values(["MLB", "value_now"], ascending=False, kind="stable").drop_duplicates("ID").copy()
    best["pitcher"] = best["slot"].isin(["SP", "RP"])
    starters = acquire.team_starters(frame, acquire.Settings())
    mine = starters[starters["org"] == str(org)]
    weakest = mine.groupby("slot")["value"].min()
    best["Gain"] = (best["value_now"] - best["slot"].map(weakest)).clip(lower=0).fillna(0.0)
    best = best.merge(minors.prospect_info(frame), on="ID", how="left")
    best["Upside"] = best["Ceil"] - best["slot"].map(
        weakest
    )  # what his ceiling would add over the weakest MLB starter at the position
    best = minors.add_tier(best, rules, version)
    flags, exact = on_forty(frame, settings)
    on40 = flags.groupby(frame["ID"]).max()
    out = pd.DataFrame(
        {
            "ID": best["ID"], "Player": (best["First Name"].astype(str) + " " + best["Last Name"].astype(str)).map(depth._tidy),
            "Pos": best["slot"], "Age": best["Age"], "Level": best["team_level"], "pitcher": best["pitcher"],
            "Value": best["value_now"], "MLB": best["MLB"], "Gain": best["Gain"], "Upside": best["Upside"],
            "Ceil": best["Ceil"], "Floor": best["Floor"],
            "Top50": best["Top50"], "ceiling_pct": best["ceiling_pct"], "Tier": best["Tier"], "Window": best["Window"],
            "focus": best["focus"], "R5": flagged(best["R5"]) if "R5" in best else False,
            "On40": best["ID"].map(on40).fillna(False).astype(bool),
            "Owed": best["SLR"].map(parse_money) if "SLR" in best else np.nan,
            "Years": best["YL"].astype(str) if "YL" in best else "",
        }
    )  # fmt: skip
    out.attrs["exact"] = exact
    return out.reset_index(drop=True)


def recommend(table: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """The draft eligible players of the table with Action (Protect, Low risk or Expose) and Why, the ones to protect first."""
    eligible = table[table["R5"]].copy()
    ready = eligible["MLB"] >= settings.ready_percentile
    useful = ready & (eligible["Gain"] > 0)  # would start over a player of the MLB lineup now
    upside = eligible["Upside"].fillna(-1) > 0  # his ceiling would start there
    top = eligible["Tier"] == minors.TOP_PROSPECT
    prospect = eligible["focus"] & (upside | top)  # a top prospect is worth protecting wherever he plays
    # a team that is rebuilding does not mind a roster spot wasted on a player who is not ready: it drafts on potential
    potential = eligible["focus"] & (eligible["ceiling_pct"].fillna(0) >= settings.potential_percentile)
    protect = useful | (prospect & (ready | top)) | potential
    low_risk = prospect & ~protect
    eligible["Action"] = np.where(protect, PROTECT, np.where(low_risk, LOW_RISK, EXPOSE))
    why = pd.Series("", index=eligible.index)
    why[prospect & ready] = "prospect and ready for the majors"
    why[prospect & top] = "top prospect"
    why[potential & ~ready & ~top] = "high potential: a rebuilding team could take him without playing him"
    why[useful] = "would start over a player of the MLB lineup" + np.where(prospect[useful], " and is a prospect", "")
    why[low_risk] = "prospect but not ready for the majors: unlikely to be picked"
    why[~protect & ~low_risk] = np.where(
        eligible.loc[~protect & ~low_risk, "focus"],
        "prospect whose ceiling would not start in the majors",
        "no prospect, would not start",
    )
    eligible["Why"] = why
    eligible["_order"] = eligible["Action"].map(ORDER)
    return eligible.sort_values(["_order", "ceiling_pct", "MLB"], ascending=[True, False, False]).drop(columns="_order")


def roster_room(table: pd.DataFrame, settings: Settings, protect: int, signings: int = 0) -> dict:
    """The 40-man arithmetic: players on it, open slots, what the plans need and how many players have to come off."""
    on = int(table["On40"].sum())
    open_slots = max(settings.size - on, 0)
    needed = protect + signings
    return {"on_forty": on, "open": open_slots, "needed": needed, "to_clear": max(needed - open_slots, 0)}


def removal_candidates(table: pd.DataFrame, starters: pd.DataFrame, org: str) -> pd.DataFrame:
    """The players on the 40-man who could go to make room, the ones with the least to keep them first: not a starter of the MLB
    lineup, rotation or bullpen, no prospect with development left, the lowest value now. `Keep` is the larger of his readiness
    and (for a prospect) his ceiling, as a percentile."""
    kept = set(starters.loc[starters["org"] == str(org), "ID"].dropna())
    forty = table[table["On40"] & ~table["ID"].isin(kept)].copy()
    forty["Keep"] = np.where(
        forty["focus"], np.maximum(forty["MLB"].fillna(0), forty["ceiling_pct"].fillna(0)), forty["MLB"].fillna(0)
    )
    return forty.sort_values(["Keep", "Value"]).reset_index(drop=True)


def draft_targets(
    frame: pd.DataFrame, org: str, settings: Settings, acquire_settings: acquire.Settings
) -> pd.DataFrame:
    """Draft eligible players of other organizations who would improve your MLB roster: his value at the position, what he adds
    over the starter of the place he would take (the biggest gain among the places), best first. A pick has to stay on the MLB
    roster all season and takes a 40-man slot."""
    needs = acquire.needs(frame, org, acquire_settings)
    internal = acquire.candidate_pool(frame, org, needs, acquire_settings)
    internal = internal[internal["Source"] == acquire.INTERNAL].groupby("slot")["value"].max()
    vacant = needs["ID"].isna()  # a place with nobody: the best of the organization's other players would fill it
    needs.loc[vacant, "value"] = np.maximum(
        needs.loc[vacant, "value"], needs.loc[vacant, "slot"].map(internal).fillna(-np.inf)
    )
    flags, _ = on_forty(frame, settings)
    taken = ~frame["org"].isin(["FA", "nan", ""]) & (frame["org"] != str(org))
    eligible = flagged(frame["R5"]) if "R5" in frame else pd.Series(False, index=frame.index)
    rows = frame[taken & eligible & ~flags & frame["slot"].isin(depth.CHART_SLOTS)].dropna(subset=["value_now"])
    pool = pd.DataFrame(
        {
            "ID": rows["ID"], "slot": rows["slot"],
            "Player": (rows["First Name"].astype(str) + " " + rows["Last Name"].astype(str)).map(depth._tidy),
            "Age": rows["Age"], "Source": "Rule 5", "From": rows["org"], "Deal": "", "Cost": 0.0, "Sign": "", "Years": "",
            "value": rows["value_now"],
        }
    )  # fmt: skip
    found = [acquire.options(pool, row, acquire_settings) for _, row in needs.iterrows()]
    found = [f for f in found if not f.empty]
    if not found:
        return pd.DataFrame()
    both = pd.concat(found, ignore_index=True).sort_values("Gain", ascending=False).drop_duplicates("ID")
    return both.reset_index(drop=True)


def potential_targets(
    frame: pd.DataFrame, org: str, settings: Settings, rules: minors.Rules, version: str = "OOTP27"
) -> pd.DataFrame:
    """Draft eligible prospects of other organizations that are not on their 40-man roster and have high potential: a ceiling
    of at least `potential_percentile` among the minor leaguers of their position, development left. Whether they would help the
    MLB roster now does not matter: a team in a weak season can take one and keep him on the 26-man roster for his potential.
    Columns ID, slot, Player, Pos, Age, From, Level, Value, Ceil, Floor, Top50, ceiling_pct, Tier, Window; the best ceiling first.
    """
    flags, _ = on_forty(frame, settings)
    taken = ~frame["org"].isin(["FA", "nan", ""]) & (frame["org"] != str(org))
    eligible = flagged(frame["R5"]) if "R5" in frame else pd.Series(False, index=frame.index)
    rows = frame[taken & eligible & ~flags & frame["slot"].isin(depth.CHART_SLOTS)].dropna(subset=["value_now"])
    if rows.empty:
        return pd.DataFrame()
    ceiling = depth.ranking_column(frame, "Ceiling (10 years)")
    best = (
        rows.dropna(subset=[ceiling]).sort_values(ceiling, ascending=False, kind="stable").drop_duplicates("ID").copy()
    )
    best["pitcher"] = best["slot"].isin(["SP", "RP"])
    best = minors.add_tier(best.merge(minors.prospect_info(frame), on="ID", how="inner"), rules, version)
    best = best[best["focus"] & (best["ceiling_pct"] >= settings.potential_percentile)]
    out = pd.DataFrame(
        {
            "ID": best["ID"], "slot": best["slot"],
            "Player": (best["First Name"].astype(str) + " " + best["Last Name"].astype(str)).map(depth._tidy),
            "Pos": best["slot"], "Age": best["Age"], "From": best["org"], "Level": best["team_level"],
            "Value": best["value_now"], "Ceil": best["Ceil"], "Floor": best["Floor"], "Top50": best["Top50"],
            "ceiling_pct": best["ceiling_pct"], "Tier": best["Tier"], "Window": best["Window"],
        }
    )  # fmt: skip
    return out.sort_values(["ceiling_pct", "Ceil"], ascending=False).reset_index(drop=True)

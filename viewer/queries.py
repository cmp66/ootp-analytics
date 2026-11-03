"""Lists and comparisons over the outlook files of a season (plan 16, stage 5).

The inputs are the two files of scripts/predict_outlook.py plus the identity columns of the season's player-data. Everything
here is a pure function of data frames, so it is tested without a UI; the viewer of plan 15 (store, filters, tables) calls the
same functions.

- `load_outlook` joins slots, players and identity and adds the pool flags (free agent, international amateur, draft class).
- `select_pool` picks a group of players: everybody, one organization, other organizations (trade targets), free agents,
  international amateurs (IAFA), the draft class.
- `tier_list` ranks players for a list: the very good players (chance of being top 10 at a position), the good ones (chance of
  being 11-50) and the ordered ranking of projected value (floor / median / ceiling blend).
- `fit_table` compares players with one organization's roster: the chance that his value in h seasons beats the value now of
  the player he would have to replace to start at the position, or to make the 26-man as a backup.
"""

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from model.outlook_scoring import QUANTILE_LEVELS, exceed_probability  # noqa: F401
from util import paths
from util.config import PROJECT_ROOT, get_save

VIEWER_CONFIG = PROJECT_ROOT / "config" / "viewer.toml"
HORIZONS = (1, 2, 3, 5, 7, 10, 99)  # 99: the career view, the best season within the next 10 (outlook_models.CAREER)
CAREER = 99
LEVEL_NAMES = {0: "FA", 1: "MLB", 2: "AAA", 3: "AA", 4: "A", 5: "A-", 6: "Rookie"}
IDENTITY_COLUMNS = [
    "ID",
    "First Name",
    "Last Name",
    "nick_name",
    "ORG",
    "TM",
    "Lev",
    "B",
    "T",
    "free_agent",
    "organization_id",
    "Draft",
    "SctCat",
    "PROY",
]
IAFA_MAX_AGE = 19  # SctCat "International" also holds foreign professionals; the IAFA prospects are the young ones
SLOT_ORDER = ["C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "DH", "SP", "RP"]


# ---------------------------------------------------------------- loading and pools


def load_outlook(save: str, season: int) -> pd.DataFrame:
    """One row per player and ranked slot, with the player-level columns, names, organization and the pool flags."""
    slots = pd.read_csv(paths.output_file(save, season, "outlook-slots"))
    players = pd.read_csv(paths.output_file(save, season, "outlook-players"))
    identity = pd.read_csv(paths.output_file(save, season, "player-data"), low_memory=False).drop_duplicates("ID")
    identity = identity[[c for c in IDENTITY_COLUMNS if c in identity.columns]]
    shared = [c for c in players.columns if c in slots.columns and c not in ("ID", "season")]
    frame = slots.merge(players.drop(columns=shared), on=["ID", "season"], how="left").merge(
        identity, on="ID", how="left"
    )
    return add_components(add_pool_flags(frame, save), save, season)


FIELDING_COLUMN = {
    "C": 2,
    "1B": 3,
    "2B": 4,
    "3B": 5,
    "SS": 6,
    "LF": 7,
    "CF": 8,
    "RF": 9,
}  # slot -> fielding model position


def add_components(frame: pd.DataFrame, save: str, season: int) -> pd.DataFrame:
    """`bat_value`, `run_value`, `field_value` (at the slot) and `pit_value`: the league-relative parts of the current value,
    from the predicted-values file of the season. Left out when that file is missing."""
    path = paths.output_file(save, season, "predicted-values")
    if not path.exists():
        return frame
    parts = pd.read_csv(path).drop_duplicates("ID").set_index("ID")
    out = frame.copy()
    for target, source in (
        ("bat_value", "batting_rel"),
        ("run_value", "baserunning_rel"),
        ("pit_value", "pitching_runs_rel"),
    ):
        if source in parts:
            out[target] = out["ID"].map(parts[source])
    field = np.full(len(out), np.nan)
    for slot, number in FIELDING_COLUMN.items():
        column = f"fielding_{number}_rel"
        if column in parts:
            mask = (out["slot"] == slot).to_numpy()
            field[mask] = out.loc[mask, "ID"].map(parts[column]).to_numpy()
    out["field_value"] = field
    pitcher = out["slot"].isin(["SP", "RP"])  # a pitcher's value is the pitching part only, a hitter's the other three
    out.loc[pitcher, ["bat_value", "run_value", "field_value"]] = np.nan
    out.loc[~pitcher, "pit_value"] = np.nan
    return out


def add_pool_flags(frame: pd.DataFrame, save: str) -> pd.DataFrame:
    """`org` (the organization as text), `level_name`, `is_free_agent`, `is_iafa`, `is_draft`.

    Export saves name the level (`Lev`: FA = free agent, INT = international amateur) and the organization (`ORG`). A dump holds
    only players who belong to an organization (plus a few free agents): it has no international amateurs and no draft class.
    """
    out = frame.copy()
    config = get_save(save)
    out["level_name"] = out["level"].map(lambda v: LEVEL_NAMES.get(int(v), "other") if pd.notna(v) else "FA")
    out["team_level"] = out["level_name"]  # the team: an export names it (A+ and A-, R+ and R), a dump only the level
    if config.source == "export":
        lev = out["Lev"].astype(str) if "Lev" in out else pd.Series("", index=out.index)
        scouting = out["SctCat"].astype(str) if "SctCat" in out else pd.Series("", index=out.index)
        out["org"] = out["ORG"].astype(str)
        out["is_draft"] = scouting == "Amateur"  # the draft class, in the exports taken before the draft
        out["is_iafa"] = (scouting == "International") & (out["Age"] <= IAFA_MAX_AGE) & (lev == "FA")
        out["is_free_agent"] = (lev == "FA") & ~out["is_draft"] & ~out["is_iafa"]
        out["level_name"] = np.where(lev.isin(["INT", "WL"]), lev, out["level_name"])
        out["team_level"] = np.where(lev != "", lev, out["level_name"])
    else:
        organization = pd.to_numeric(out.get("organization_id", 0), errors="coerce").fillna(0)
        out["org"] = organization.astype(int).astype(str)
        out["is_free_agent"] = organization <= 0
        out["is_iafa"] = False
        out["is_draft"] = False
    return out


def select_pool(frame: pd.DataFrame, pool: str, org: str | None = None) -> pd.DataFrame:
    """Players of a pool: all, mlb, org (my organization), others (trade targets), all_orgs (every organization), free_agents,
    iafa, draft."""
    if pool in ("org", "others") and org is None:
        raise ValueError(f"pool {pool!r} needs an organization")
    masks = {
        "all": pd.Series(True, index=frame.index),
        "mlb": frame["level"] == 1,
        "org": frame["org"] == str(org),
        "others": (frame["org"] != str(org)) & ~frame["is_free_agent"] & ~frame["is_iafa"],
        "all_orgs": ~frame["org"].isin(["FA", "nan", ""]),  # everybody who belongs to an organization
        "free_agents": frame["is_free_agent"],
        "iafa": frame["is_iafa"],
        "draft": frame["is_draft"],
    }
    if pool not in masks:
        raise ValueError(f"unknown pool {pool!r}; known: {sorted(masks)}")
    return frame[masks[pool]]


def apply_filters(
    frame: pd.DataFrame, slots=None, age_min=None, age_max=None, levels=None, include_depth=True
) -> pd.DataFrame:
    """Slots, age range, level names (MLB, AAA, ...) and whether to keep the depth players."""
    mask = pd.Series(True, index=frame.index)
    if slots:
        mask &= frame["slot"].isin(slots)
    if age_min is not None:
        mask &= frame["Age"] >= age_min
    if age_max is not None:
        mask &= frame["Age"] <= age_max
    if levels:
        mask &= frame["level_name"].isin(levels)
    if not include_depth:
        mask &= ~frame["depth_player"].astype(bool)
    return frame[mask]


# ---------------------------------------------------------------- lists


def available_horizons(frame: pd.DataFrame) -> tuple[int, ...]:
    """The horizons whose columns are in the outlook frame (files made before the longer horizons have fewer)."""
    return tuple(h for h in HORIZONS if f"rank_score_h{h}" in frame.columns)


def metric_column(metric: str, horizon: int, frame: pd.DataFrame) -> pd.Series:
    """The sort key of a list: p_top10, p_11_50 (top 50 but not top 10), p_top50, rank_score, value, floor, ceiling,
    season_value, p_start or p_roster."""
    if metric == "p_11_50":
        return (frame[f"p_top50_h{horizon}"] - frame[f"p_top10_h{horizon}"]).clip(lower=0)
    name = f"{metric}_h{horizon}"
    if name not in frame.columns:
        raise KeyError(
            f"no column {name}; metrics: p_top10, p_11_50, p_top50, rank_score, value, floor, ceiling, season_value, "
            "p_start, p_roster"
        )
    return frame[name]


def tier_list(frame: pd.DataFrame, metric: str, horizon: int, top: int = 50, by_player: bool = False) -> pd.DataFrame:
    """Rows sorted by a metric (best first), with the list's key as `key`; `by_player` keeps one row per player."""
    ranked = frame.assign(key=metric_column(metric, horizon, frame).to_numpy())
    ranked = ranked.sort_values(["key", f"rank_score_h{horizon}"], ascending=False)
    if by_player:
        ranked = ranked.drop_duplicates("ID")
    return ranked.head(top).reset_index(drop=True)


# ---------------------------------------------------------------- display


SHOWN = [
    "First Name", "Last Name", "Age", "org", "level_name", "slot", "value_now", "tier_now",
    "bat_value", "run_value", "field_value", "pit_value",
    "floor", "value", "ceiling", "p_top10", "p_top50", "p_stay", "season_value", "injury_cost", "p_start", "p_roster",
]  # fmt: skip


def display_table(rows: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """The columns of a list with the horizon's columns named plainly."""
    out = pd.DataFrame(index=rows.index)
    for column in SHOWN:
        for source in (column, f"{column}_h{horizon}"):
            if source in rows.columns:
                out[column] = rows[source]
                break
    out.insert(0, "key", rows["key"].round(3))
    return out


# ---------------------------------------------------------------- comparison with an organization


def quantile_matrix(frame: pd.DataFrame, horizon: int) -> np.ndarray:
    return frame[
        [f"floor_h{horizon}", f"q25_h{horizon}", f"value_h{horizon}", f"q75_h{horizon}", f"ceiling_h{horizon}"]
    ].to_numpy()


@dataclass(frozen=True)
class DepthRules:
    """How many players of an organization hold a slot: starters (the player to beat to start) and roster (to make the 26-man)."""

    starters: dict = field(
        default_factory=lambda: {
            "C": 1,
            "1B": 1,
            "2B": 1,
            "3B": 1,
            "SS": 1,
            "LF": 1,
            "CF": 1,
            "RF": 1,
            "DH": 1,
            "SP": 5,
            "RP": 7,
        }
    )
    roster_extra: dict = field(
        default_factory=lambda: {
            "C": 1,
            "1B": 1,
            "2B": 1,
            "3B": 1,
            "SS": 1,
            "LF": 1,
            "CF": 1,
            "RF": 1,
            "DH": 1,
            "SP": 1,
            "RP": 2,
        }
    )
    roster_levels: tuple = (1,)  # MLB level = the 26-man in an export


def org_thresholds(frame: pd.DataFrame, org: str, rules: DepthRules = DepthRules()) -> pd.DataFrame:
    """Per slot, the value now of the organization's k-th best player there (k = starters for `start`, + extra for `roster`).

    Only players at the roster levels count (MLB by default). With fewer than k players ranked at the slot the threshold is -inf.
    """
    mine = frame[(frame["org"] == str(org)) & frame["level"].isin(rules.roster_levels)]
    rows = []
    for slot in SLOT_ORDER:
        values = mine.loc[mine["slot"] == slot, "value_now"].sort_values(ascending=False).to_numpy()
        k_start = rules.starters.get(slot, 1)
        k_roster = k_start + rules.roster_extra.get(slot, 1)
        pick = lambda k: float(values[k - 1]) if len(values) >= k else -np.inf  # noqa: E731
        rows.append(
            {"slot": slot, "players": len(values), "start_threshold": pick(k_start), "roster_threshold": pick(k_roster)}
        )
    return pd.DataFrame(rows)


def fit_table(
    frame: pd.DataFrame,
    org: str,
    horizon: int,
    rules: DepthRules = DepthRules(),
    thresholds_from: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Adds `start_threshold`, `roster_threshold`, `p_start_h`, `p_roster_h` to the rows of `frame` against organization `org`.

    p_start = chance he is still in the league and his value in h seasons is above the value now of the org's k-th best player
    at the slot; p_roster the same against the org's (starters + extra)-th best (a backup's place). Values are ratings-based,
    so this is a depth-chart approximation, not the lineup. The org's players come from `thresholds_from` (default `frame`),
    which lets a filtered list be compared with an organization that is not in it."""
    thresholds = org_thresholds(frame if thresholds_from is None else thresholds_from, org, rules).set_index("slot")
    out = frame.join(thresholds[["start_threshold", "roster_threshold"]], on="slot")
    q = quantile_matrix(out, horizon)
    stay = out[
        f"p_stay_h{1 if horizon == CAREER else horizon}"
    ].to_numpy()  # the career view has no stay model of its own
    out[f"p_start_h{horizon}"] = stay * exceed_probability(q, out["start_threshold"].to_numpy())
    out[f"p_roster_h{horizon}"] = stay * exceed_probability(q, out["roster_threshold"].to_numpy())
    return out


# ---------------------------------------------------------------- list definitions (config/viewer.toml)


@dataclass(frozen=True)
class ListSpec:
    name: str
    title: str
    metric: str
    pool: str = "all"
    slots: tuple = ()
    age_min: int | None = None
    age_max: int | None = None
    levels: tuple = ()
    include_depth: bool = True
    by_player: bool = False


def load_list_specs(path: Path = VIEWER_CONFIG) -> tuple[dict, dict[str, ListSpec]]:
    """(defaults, lists) from config/viewer.toml."""
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    defaults = dict(raw.get("defaults", {}))
    lists = {}
    for name, values in raw.get("lists", {}).items():
        lists[name] = ListSpec(
            name=name,
            title=values.get("title", name),
            metric=values["metric"],
            pool=values.get("pool", "all"),
            slots=tuple(values.get("slots", ())),
            age_min=values.get("age_min"),
            age_max=values.get("age_max"),
            levels=tuple(values.get("levels", ())),
            include_depth=values.get("include_depth", True),
            by_player=values.get("by_player", False),
        )
    return defaults, lists


def run_list(
    frame: pd.DataFrame, spec: ListSpec, horizon: int, org: str | None = None, top: int = 50, slots=None
) -> pd.DataFrame:
    """A configured list for one organization (when the pool or the metric needs one) and horizon."""
    pool = select_pool(frame, spec.pool, org)
    pool = apply_filters(pool, slots or spec.slots, spec.age_min, spec.age_max, spec.levels, spec.include_depth)
    if spec.metric in ("p_start", "p_roster"):
        if org is None:
            raise ValueError(f"list {spec.name!r} compares with an organization: give one")
        pool = fit_table(pool, org, horizon, thresholds_from=frame)
    return tier_list(pool, spec.metric, horizon, top, spec.by_player)

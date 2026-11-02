"""Longitudinal tables for the outlook models (plan 16, stage 1): who each player is, how much he played, what his
ratings are worth at each position, and where he stands among the players of the league, one season at a time.

Two tables per save are written to files/<save>/outlook/:
- players.csv.gz: one row per player and season: identity, traits (WE, INT, leadership), the four prone fields, the
  entry route and season, level, playing time at every level, the gate ratings, and whether he is still in the league 1,
  2, 3 and 5 seasons later (`stay_h<n>`, `observed_h<n>`).
- slot_values.csv.gz: the value at every slot for every player and season, with the flag `qualified` (the targets of the
  outlook models need the value of a player who no longer passes a gate).
- values.csv.gz: one row per player, season and slot (C, 1B, ..., RF, DH, SP, RP) that he qualifies for: the
  league-relative value of his ratings at that slot, his rank among the qualified players of the league and the tier
  (top 10, top 50, rest; ranks are for a 30-team league and scale with the number of teams).

Value of a slot = what the plan 14 models say the ratings are worth, never the stats or the playing time (a usage label
would record what the manager did, see plans/16-development-models.md): hitters at a position are batting + baserunning +
fielding at the position per season, DH is batting + baserunning, SP and RP are the pitching runs per 200 IP. All are
the league-relative (`_rel`) columns of model/relative.py. A player qualifies for a position when his key ratings reach
the position's minimum or his potential rating at the position reaches the potential minimum; the minimums are measured
(lowest ratings among the top-50 starters of the position in the long league) and stored in config/outlook.toml.
"""

import tomllib
from pathlib import Path

import numpy as np
import pandas as pd

from model import data, domains, formats, registry
from util import paths
from util.config import PROJECT_ROOT, get_save

OUTLOOK_CONFIG = PROJECT_ROOT / "config" / "outlook.toml"
HORIZONS = (1, 2, 3, 5)
TOP_RANKS = {"top10": 10, "top50": 50}  # for a 30-team league
BASE_TEAMS = 30

PITCHER_LPOS = {"P", "SP", "RP", "CL"}
POSITIONS = {2: "C", 3: "1B", 4: "2B", 5: "3B", 6: "SS", 7: "LF", 8: "CF", 9: "RF"}
HITTER_SLOTS = [*POSITIONS.values(), "DH"]
PITCHER_SLOTS = ["SP", "RP"]
# key ratings of each gated slot (plans/16: from the plan 14 attribute review)
GATE_KEYS = {
    "C": ["C FRM", "C ABI"],
    "1B": ["IF RNG"],
    "2B": ["IF RNG", "IF ARM"],
    "3B": ["IF RNG", "IF ARM"],
    "SS": ["IF RNG", "IF ARM"],
    "LF": ["OF RNG"],
    "CF": ["OF RNG"],
    "RF": ["OF RNG", "OF ARM"],
    "SP": ["STM", "PIT"],
}
POTENTIAL = "pot"  # gate key of the potential rating of the position
GATE_PERCENTILE = 0.01

STATE_COLUMNS = [
    "Age", "LPOS", "role", "ORG", "league_id", "free_agent", "retired", "last_team_id", "B", "T", "HT", "WT",
    "WE", "INT", "LEA", "LOY", "FIN", "Prone", "prone_leg", "prone_back", "prone_arm", "experience", "college",
    "acquired", "draft_year", "draft_round", "draft_overall_pick", "picked_in_draft", "draft_eligible",
    "injury_is_injured", "injury_left", "injury_career_ending",
]  # fmt: skip
POSITION_RATINGS = list(POSITIONS.values())
POTENTIAL_RATINGS = [f"{name} Pot" for name in POSITIONS.values()]
GATE_RATINGS = sorted({key for keys in GATE_KEYS.values() for key in keys})
TEXT_COLUMNS = {"LPOS", "role", "B", "T", *(f"{t} cat" for t in (*formats.TRAITS, formats.PRONE))}


# ---------------------------------------------------------------- small pure helpers


def tier_cutoffs(teams: int) -> dict[str, int]:
    """Rank cutoffs of the tiers for a league with this many teams (10 and 50 for 30 teams)."""
    return {name: max(1, round(rank * teams / BASE_TEAMS)) for name, rank in TOP_RANKS.items()}


def entry_route(frame: pd.DataFrame) -> pd.Series:
    """How a player entered, from his first row: drafted out of college, drafted out of high school, signed
    internationally (undrafted at 17 or younger) or other (undrafted older players)."""
    drafted = frame["picked_in_draft"].fillna(0) == 1
    college = frame["college"].fillna(0) == 1
    route = np.where(
        drafted & college,
        "college_draft",
        np.where(drafted, "hs_draft", np.where(frame["Age"] <= 17, "international", "other")),
    )
    return pd.Series(route, index=frame.index)


def floor_to_five(value) -> int:
    return int(np.floor(float(value) / 5.0) * 5)


def _floor(rating: str, value) -> int:
    """Floor to a multiple of 5 (20-80 ratings); the number of pitches is a small count and floors to an integer."""
    return int(np.floor(float(value))) if rating == "PIT" else floor_to_five(value)


def team_count(save: str, season: int) -> int:
    """Number of MLB teams in the league that season: the major league teams of the team batting table (dump saves hold
    every level, plus all-star squads that have a fraction of a season's plate appearances)."""
    teams = data.read_output(save, season, "team-batting")
    if "LEVEL_ID" in teams:
        teams = teams[teams["LEVEL_ID"] == 1]
    return int((teams["PA"] >= 0.5 * teams["PA"].median()).sum())


# ---------------------------------------------------------------- per season state


def league_levels(save: str, season: int) -> pd.Series:
    """Level (1 = MLB, 2 = AAA, ...) of every league from the dump's teams.csv, indexed by league_id; empty for exports.

    A player's `league_id` is the league he plays in now (0 for a free agent); `last_team_id` is his parent club."""
    config = get_save(save)
    if config.source != "dump":
        return pd.Series(dtype=float)
    path = paths.dump_dir(save, config.ratings, season) / "teams.csv"
    if not path.exists():
        return pd.Series(dtype=float)
    teams = pd.read_csv(path, usecols=["league_id", "level"])
    return teams.groupby("league_id")["level"].min()


def season_state(save: str, season: int) -> pd.DataFrame:
    """Identity, traits, entry fields, level and gate ratings of every player of a season, indexed by ID."""
    players = data.read_output(save, season, "player-data").drop_duplicates("ID").set_index("ID")
    categories = [f"{t} cat" for t in (*formats.TRAITS, formats.PRONE)]
    wanted = [*STATE_COLUMNS, *POSITION_RATINGS, *POTENTIAL_RATINGS, *GATE_RATINGS, *categories]
    state = players[[c for c in dict.fromkeys(wanted) if c in players.columns]].copy()
    numeric = [c for c in state.columns if c not in TEXT_COLUMNS]
    state[numeric] = state[numeric].apply(pd.to_numeric, errors="coerce")
    state.insert(0, "season", season)
    state["level"] = state["league_id"].map(league_levels(save, season)) if "league_id" in state else np.nan
    if "free_agent" in state:
        state["level"] = state["level"].where(state["free_agent"] != 1)  # a free agent is between teams
    if get_save(save).source == "export" and "Lev" in players.columns:  # the export names the level instead
        state["level"] = players["Lev"].map(lambda value: formats.export_level_number(value, get_save(save).ratings))
    return state


def _stat_rows(folder: Path, name: str, columns: list[str], season: int) -> pd.DataFrame:
    path = folder / f"players_career_{name}_stats.csv"
    if not path.exists():
        return pd.DataFrame(columns=columns)
    rows = pd.read_csv(path, usecols=columns)
    return rows[(rows["year"] == season) & (rows["split_id"] == 1)]


def playing_time(save: str, season: int) -> pd.DataFrame:
    """Playing time of a season at every level (dump saves), indexed by player ID.

    pa and bf are the totals over all levels, `primary_level` is the level with the most of them, mlb_pa / mlb_bf the
    MLB part. `time_share` compares the volume with the 90th percentile of players whose main level and role
    (starter or not, for pitchers) is the same: 1 is a full season for that level, lower values mean missed time
    (injury, call-up, demotion). It is a proxy: a player without stats gets NaN."""
    config = get_save(save)
    empty = pd.DataFrame(columns=["pa", "bf", "primary_level", "mlb_pa", "mlb_bf", "time_share"])
    empty.index.name = "ID"
    if config.source != "dump":
        return empty
    folder = paths.dump_dir(save, config.ratings, season)
    bat = _stat_rows(folder, "batting", ["player_id", "year", "level_id", "split_id", "pa", "g"], season)
    pit = _stat_rows(folder, "pitching", ["player_id", "year", "level_id", "split_id", "bf", "g", "gs"], season)
    if bat.empty and pit.empty:
        return empty
    by_level = lambda frame, column: frame.groupby(["player_id", "level_id"])[column].sum().reset_index()  # noqa: E731
    bat_level, pit_level = by_level(bat, "pa"), by_level(pit, "bf")

    def main_level(frame, column):
        top = frame.sort_values(column, ascending=False).drop_duplicates("player_id")
        return top.set_index("player_id")["level_id"]

    out = pd.DataFrame(index=pd.Index(sorted(set(bat["player_id"]) | set(pit["player_id"])), name="ID"))
    out["pa"] = bat_level.groupby("player_id")["pa"].sum().reindex(out.index).fillna(0)
    out["bf"] = pit_level.groupby("player_id")["bf"].sum().reindex(out.index).fillna(0)
    bat_main, pit_main = main_level(bat_level, "pa"), main_level(pit_level, "bf")
    out["primary_level"] = np.where(out["pa"] >= out["bf"], bat_main.reindex(out.index), pit_main.reindex(out.index))
    out["mlb_pa"] = bat_level[bat_level["level_id"] == 1].groupby("player_id")["pa"].sum().reindex(out.index).fillna(0)
    out["mlb_bf"] = pit_level[pit_level["level_id"] == 1].groupby("player_id")["bf"].sum().reindex(out.index).fillna(0)
    games = pit.groupby("player_id")[["g", "gs"]].sum().reindex(out.index).fillna(0)
    out["starter"] = (games["gs"] >= 0.5 * games["g"]) & (games["g"] > 0)
    out["volume"] = np.where(out["pa"] >= out["bf"], out["pa"], out["bf"])
    out["batter"] = out["pa"] >= out["bf"]
    return _with_time_share(out)


def _with_time_share(out: pd.DataFrame, min_volume: int = 50) -> pd.DataFrame:
    keys = ["batter", "primary_level", "starter"]
    group_key = out.assign(starter=out["starter"] & ~out["batter"])  # starter only matters for pitchers
    enough = group_key[group_key["volume"] >= min_volume]
    reference = enough.groupby(keys)["volume"].quantile(0.9).rename("reference")
    joined = group_key.join(reference, on=keys)
    out["time_share"] = (joined["volume"] / joined["reference"]).clip(upper=1.0).to_numpy()
    return out.drop(columns=["volume", "batter", "starter"])


# ---------------------------------------------------------------- values per slot


def slot_values(state: pd.DataFrame, scored: pd.DataFrame) -> pd.DataFrame:
    """Long table (ID, slot, value) of the league-relative value of every player at every slot, before the gates."""
    pitcher = state["LPOS"].isin(PITCHER_LPOS)
    frames = []
    hitters = state.index[~pitcher]
    bat_run = scored["batting_rel"].reindex(hitters) + scored["baserunning_rel"].reindex(hitters)
    frames.append(pd.DataFrame({"ID": hitters, "slot": "DH", "value": bat_run.to_numpy()}))
    for position, name in POSITIONS.items():
        field = scored[f"fielding_{position}_rel"].reindex(hitters)
        frames.append(pd.DataFrame({"ID": hitters, "slot": name, "value": (bat_run + field).to_numpy()}))
    arms = state.index[pitcher]
    runs = scored["pitching_runs_rel"].reindex(arms).to_numpy()
    for slot in PITCHER_SLOTS:
        frames.append(pd.DataFrame({"ID": arms, "slot": slot, "value": runs}))
    values = pd.concat(frames, ignore_index=True)
    return values.dropna(subset=["value"])


# ---------------------------------------------------------------- gates


def qualifies(state: pd.DataFrame, slot: str, gates: dict) -> pd.Series:
    """Whether each player (indexed by ID) qualifies for a slot: key ratings at the minimum, or for a position the
    potential rating at the potential minimum. DH and RP have no gate."""
    if slot not in GATE_KEYS:
        return pd.Series(True, index=state.index)
    minimums = gates[slot]
    current = pd.Series(True, index=state.index)
    for key in GATE_KEYS[slot]:
        current &= state[key] >= minimums[key]
    if slot in POSITIONS.values() and POTENTIAL in minimums:
        current |= state[f"{slot} Pot"] >= minimums[POTENTIAL]
    return current.fillna(False)


def gate_sample(save: str, season: int, state: pd.DataFrame, values: pd.DataFrame, teams: int) -> dict:
    """Ratings of the top-50 MLB starters of each gated slot in one season (the sample the minimums are measured on).

    Starters are the players with the playing time of a regular at the position (fielding table) or of a starting
    pitcher (usage bucket SP / SP-lean); the top 50 are ranked by the slot value."""
    top = tier_cutoffs(teams)["top50"]
    sample = {}
    fielding = data.read_output(save, season, "fielding")
    fielding = fielding[fielding["IPClean"] >= domains.MIN_FIELDING_IP]
    pitching = data.read_output(save, season, "pitching")
    pitching = pitching[(pitching["IPClean"] >= domains.MIN_IP) & pitching["PBaseline"].isin(["SP", "SP-lean"])]
    for slot in GATE_KEYS:
        if slot == "SP":
            starters = set(pitching["ID"])
        else:
            position = next(p for p, name in POSITIONS.items() if name == slot)
            starters = set(fielding.loc[fielding["POS"] == position, "ID"])
        rows = values[(values["slot"] == slot) & values["ID"].isin(starters)].nlargest(top, "value")
        columns = [*GATE_KEYS[slot], *([f"{slot} Pot"] if slot in POSITIONS.values() else [])]
        sample[slot] = state.reindex(rows["ID"])[columns]
    return sample


def gates_from_samples(samples: list[dict]) -> dict:
    """Minimum rating per gated slot: the 1st percentile of the top-50 starters, floored to a multiple of 5."""
    gates = {}
    for slot in GATE_KEYS:
        pooled = pd.concat([sample[slot] for sample in samples if slot in sample], ignore_index=True)
        minimums = {key: _floor(key, pooled[key].dropna().quantile(GATE_PERCENTILE)) for key in GATE_KEYS[slot]}
        if slot in POSITIONS.values():
            minimums[POTENTIAL] = floor_to_five(pooled[f"{slot} Pot"].dropna().quantile(GATE_PERCENTILE))
        gates[slot] = minimums
    return gates


def _read_config() -> dict:
    if not OUTLOOK_CONFIG.exists():
        return {}
    with open(OUTLOOK_CONFIG, "rb") as f:
        return tomllib.load(f)


def _write_config(config: dict) -> None:
    """Write config/outlook.toml: per ratings set the note, the trait cut points and the position minimums."""
    lines = [
        "# Settings of the outlook tables (plans/16-development-models.md), per ratings set. Written by",
        "# scripts/build_outlook_tables.py and edited by hand where needed.",
        "# Slot tables: minimum ratings to be ranked at a slot, the 1st percentile of the top-50 MLB starters of the slot",
        "# floored to a multiple of 5. A player qualifies when all key ratings reach the minimum, or (positions) the",
        "# potential rating reaches `pot`.",
        "# [<ratings>.traits]: upper bounds of the dump's 1-200 numbers for each category, estimated by matching the shares",
        "# of the in-game export (model/formats.py); the last category has no bound.",
    ]
    for name, values in config.items():
        lines += ["", f"[{name}]", f'measured = "{values.get("measured", "")}"']
        for slot, minimums in values.items():
            if slot in GATE_KEYS:
                lines.append(f"[{name}.{slot}]")
                lines += [f'"{k}" = {v}' if k != POTENTIAL else f"{k} = {v}" for k, v in minimums.items()]
        if values.get("traits"):
            lines += [f"[{name}.traits]", f'measured = "{values["traits"].get("measured", "")}"']
            lines += [f"{t} = {b}" for t, b in values["traits"].items() if t != "measured"]
    OUTLOOK_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    OUTLOOK_CONFIG.write_text("\n".join(lines) + "\n")


def load_gates(ratings: str) -> dict | None:
    """The minimums of a ratings set from config/outlook.toml (None when not measured yet)."""
    found = _read_config().get(ratings)
    gates = {slot: dict(minimums) for slot, minimums in (found or {}).items() if slot in GATE_KEYS}
    return gates or None


def write_gates(ratings: str, gates: dict, note: str) -> None:
    """Write the minimums of a ratings set into config/outlook.toml, keeping everything else in it."""
    config = _read_config()
    entry = config.setdefault(ratings, {})
    for slot in [s for s in entry if s in GATE_KEYS]:
        del entry[slot]
    entry.update({slot: dict(minimums) for slot, minimums in gates.items()})
    entry["measured"] = note
    _write_config(config)


def load_trait_bounds(ratings: str) -> dict | None:
    """Cut points of the trait categories for a ratings set: {trait: [bounds]} (None when not fitted yet)."""
    traits = (_read_config().get(ratings) or {}).get("traits")
    return {t: list(b) for t, b in traits.items() if t != "measured"} if traits else None


def write_trait_bounds(ratings: str, bounds: dict, note: str) -> None:
    config = _read_config()
    config.setdefault(ratings, {})["traits"] = {**bounds, "measured": note}
    _write_config(config)


# ---------------------------------------------------------------- traits: dump numbers and export categories


def trait_labels(trait: str):
    """Categories of a trait, lowest to highest. Proneness is used in three levels: Iron Man (sustained 20 or less) and
    Wrecked (sustained 180 or more) depend on history an export does not carry, so they join Durable and Fragile."""
    return formats.PRONE3_LABELS if trait == formats.PRONE else formats.TRAIT_LABELS


def export_trait_shares(ratings: str) -> dict[str, dict[str, float]]:
    """Share of each category per trait over the processed export seasons of the saves with this ratings set.

    Needs the `<trait> cat` columns that the export import now keeps (process the export saves again if missing)."""
    from util.config import load_saves
    from util.seasons import available_seasons

    frames = []
    for name, save in load_saves().items():
        if save.source != "export" or save.ratings != ratings:
            continue
        for season in available_seasons(name):
            path = paths.output_file(name, season, "player-data")
            if path.exists():
                header = pd.read_csv(path, nrows=0).columns
                wanted = [f"{t} cat" for t in (*formats.TRAITS, formats.PRONE)]
                if all(c in header for c in wanted):
                    frames.append(pd.read_csv(path, usecols=wanted, low_memory=False))
    if not frames:
        raise FileNotFoundError(
            f"no processed export season with the trait category columns for ratings {ratings}; "
            "run scripts/process_seasons.py for the export saves of that ratings set"
        )
    pooled = pd.concat(frames, ignore_index=True)
    pooled[f"{formats.PRONE} cat"] = pooled[f"{formats.PRONE} cat"].map(formats.PRONE3)
    return {t: formats.label_shares(pooled[f"{t} cat"], trait_labels(t)) for t in (*formats.TRAITS, formats.PRONE)}


def fit_trait_bounds(frame: pd.DataFrame, shares: dict) -> dict[str, list[float]]:
    """Cut points of every trait from the dump numbers (a frame with the trait columns) and the export shares."""
    return {t: formats.fit_bounds(frame[t], shares[t], trait_labels(t)) for t in shares}


def with_trait_categories(frame: pd.DataFrame, bounds: dict) -> pd.DataFrame:
    """Add `<trait>_cat` and `<trait>_ord` (0 = lowest category) for the personality traits and `Prone3_cat` / `Prone3_ord`
    for proneness (durable / normal / fragile). Export frames already hold the categories in `<trait> cat`."""
    out = frame.copy()
    for trait in (*formats.TRAITS, formats.PRONE):
        labels = trait_labels(trait)
        if f"{trait} cat" in out.columns:
            category = out[f"{trait} cat"]
            if trait == formats.PRONE:
                category = category.map(formats.PRONE3)
        else:
            category = formats.categorize(pd.to_numeric(out[trait], errors="coerce"), bounds[trait], labels)
        name = "Prone3" if trait == formats.PRONE else trait
        out[f"{name}_cat"] = category
        out[f"{name}_ord"] = formats.ordinal(category, labels)
    return out.drop(columns=[c for c in out.columns if c.endswith(" cat")])


# ---------------------------------------------------------------- ranks and tiers


def rank_slots(values: pd.DataFrame, teams: int) -> pd.DataFrame:
    """Rank (1 = best), pool size, percentile and tier of every row among the rows of its slot (one season)."""
    cutoffs = tier_cutoffs(teams)
    out = values.copy()
    grouped = out.groupby("slot")["value"]
    out["rank"] = grouped.rank(ascending=False, method="first").astype(int)
    out["pool"] = grouped.transform("size")
    out["pct"] = 1 - (out["rank"] - 1) / out["pool"]
    out["tier"] = np.where(
        out["rank"] <= cutoffs["top10"], "top10", np.where(out["rank"] <= cutoffs["top50"], "top50", "rest")
    )
    return out


# ---------------------------------------------------------------- longitudinal labels


def player_history(states: pd.DataFrame, first_season: int) -> pd.DataFrame:
    """Per player: entry season and age, route, and whether the history before the table starts is unknown."""
    ordered = states.reset_index().sort_values(["ID", "season"])
    first = ordered.drop_duplicates("ID").set_index("ID")
    history = pd.DataFrame(
        {
            "entry_season": first["season"],
            "entry_age": first["Age"],
            "entry_route": entry_route(first),
            "left_truncated": first["season"] == first_season,
            "last_season": ordered.groupby("ID")["season"].max(),
        }
    )
    return history


def stay_flags(states: pd.DataFrame, last_season: int, horizons=HORIZONS) -> pd.DataFrame:
    """For each player-season and horizon h: observed_h (season + h is inside the table) and stay_h (he has a row then).

    stay_h is NaN when the future is not observed, never False: censored rows are not 'left the league'."""
    keys = pd.MultiIndex.from_frame(states.reset_index()[["ID", "season"]])
    present = set(keys)
    flags = pd.DataFrame(index=keys)
    for h in horizons:
        observed = np.array([season + h <= last_season for _, season in keys])
        stay = np.array([(player, season + h) in present for player, season in keys])
        flags[f"observed_h{h}"] = observed
        flags[f"stay_h{h}"] = np.where(observed, stay, np.nan)
    return flags


def future_slot_values(values: pd.DataFrame, horizon: int, last_season: int) -> pd.DataFrame:
    """The slot rows of season t+h joined onto the rows of season t (value, rank, tier of the same slot h seasons later).

    A row whose future season is inside the table but has no matching row (he left the league or no longer
    qualifies) gets in_pool_h = False; a row whose future is beyond the table has observed_h = False and NaN columns."""
    later = values[["ID", "slot", "season", "value", "rank", "tier"]].copy()
    later["season"] = later["season"] - horizon
    later = later.rename(columns={"value": f"value_h{horizon}", "rank": f"rank_h{horizon}", "tier": f"tier_h{horizon}"})
    out = values[["ID", "slot", "season"]].merge(later, on=["ID", "slot", "season"], how="left")
    out[f"observed_h{horizon}"] = out["season"] + horizon <= last_season
    out[f"in_pool_h{horizon}"] = out[f"value_h{horizon}"].notna()
    return out


# ---------------------------------------------------------------- build


def build_tables(save: str, seasons, models_from: str | None = None, remeasure: bool = False, log=print) -> dict:
    """Players and values tables for the seasons of a save (see the module docstring). Returns them in a dict and
    writes the gates to config/outlook.toml when they are measured."""
    config = get_save(save)
    seasons = [int(s) for s in seasons]
    models_from = models_from or save
    artifacts = _load_artifacts(models_from, config.ratings)
    states, raw_values, teams, samples = {}, {}, {}, []
    for season in seasons:
        state = season_state(save, season).join(playing_time(save, season), how="left")
        scored, _ = registry.score_season(save, season, artifacts)
        values = slot_values(state, scored)
        teams[season] = team_count(save, season)
        states[season], raw_values[season] = state, values
        samples.append(gate_sample(save, season, state, values, teams[season]))
        log(f"{save} {season}: {len(state)} players, {len(values)} slot values")
    gates = None if remeasure else load_gates(config.ratings)
    if gates is None:
        gates = gates_from_samples(samples)
        write_gates(config.ratings, gates, f"measured on {save} {seasons[0]}-{seasons[-1]}")
        log(f"measured gates for {config.ratings} -> {OUTLOOK_CONFIG}")
    ranked, every = [], []
    for season in seasons:
        state, values = states[season], raw_values[season]
        keep = []
        for slot, group in values.groupby("slot"):
            ok = group["ID"].map(qualifies(state, slot, gates)).fillna(False).astype(bool)
            keep.append(group[ok.to_numpy()])
            every.append(group.assign(season=season, qualified=ok.to_numpy()))
        qualified = pd.concat(keep, ignore_index=True)
        ranked.append(rank_slots(qualified, teams[season]).assign(season=season))
    all_states = pd.concat(states.values())
    bounds = None if remeasure else load_trait_bounds(config.ratings)
    if bounds is None and config.source == "dump":
        bounds = fit_trait_bounds(all_states, export_trait_shares(config.ratings))
        write_trait_bounds(
            config.ratings, bounds, f"fitted on {save} {seasons[0]}-{seasons[-1]} by matching export shares"
        )
        log(f"fitted trait cut points for {config.ratings} -> {OUTLOOK_CONFIG}")
    all_states = with_trait_categories(all_states, bounds or {})
    history = player_history(all_states, seasons[0])
    flags = stay_flags(all_states, seasons[-1])
    players = all_states.reset_index().merge(history, left_on="ID", right_index=True, how="left")
    players = players.merge(flags, left_on=["ID", "season"], right_index=True, how="left")
    players["years_in_system"] = np.where(
        players["left_truncated"], np.nan, players["season"] - players["entry_season"]
    )
    players["teams"] = players["season"].map(teams)
    return {
        "players": players,
        "values": pd.concat(ranked, ignore_index=True),
        "slot_values": pd.concat(every, ignore_index=True),
        "gates": gates,
    }


def _load_artifacts(models_from: str, ratings: str) -> dict:
    from model import artifact

    folder = artifact.models_dir(models_from, ratings)
    found = {}
    for path in sorted(folder.glob("*.joblib")) if folder.exists() else []:
        loaded = artifact.load_artifact(models_from, ratings, path.stem)
        found[loaded.name] = loaded
    if not found:
        raise FileNotFoundError(f"no trained models in {folder}; run scripts/train_models.py {models_from}")
    return found


def outlook_dir(save: str, create: bool = False) -> Path:
    path = paths.save_dir(save) / "outlook"
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def write_tables(save: str, tables: dict) -> list[Path]:
    folder = outlook_dir(save, create=True)
    written = []
    for name in ("players", "values", "slot_values"):
        path = folder / f"{name}.csv.gz"
        tables[name].round(4).to_csv(path, index=False)
        written.append(path)
    return written


def read_tables(save: str) -> dict:
    folder = outlook_dir(save)
    missing = [n for n in ("players", "values") if not (folder / f"{n}.csv.gz").exists()]
    if missing:
        raise FileNotFoundError(f"{folder} has no {missing}; run scripts/build_outlook_tables.py {save}")
    tables = {n: pd.read_csv(folder / f"{n}.csv.gz", low_memory=False) for n in ("players", "values")}
    extra = folder / "slot_values.csv.gz"  # every slot value, gated or not (the targets of the outlook models)
    tables["slot_values"] = pd.read_csv(extra, low_memory=False) if extra.exists() else None
    return tables

"""Data of the organization overview screen (plan 15): pure functions over the outlook frame, no UI.

The screen shows one organization on the current value of each player (`value_now`: runs above the league's average starter at the
position, from his ratings now; offense + baserunning + defense add up to it):
the best position players, the best players at each position, and the best starting and relief pitchers.
"""

import pandas as pd

from viewer import queries

HITTER_SLOTS = ["C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "DH"]
TOP_HITTERS = 20
TOP_PER_POSITION = 10
TOP_STARTERS = 20
TOP_RELIEVERS = 20


# What the long and position tables show. "current": the value now (offense + defense + baserunning for a hitter). "ceiling" and
# "floor" (minor league page): the best 90th / 10th percentile value he reaches in 1, 2, 3 or 5 seasons ("five"), or the 90th /
# 10th percentile of his best season within the next 10 ("career"), and the matching chance of becoming a top-10 (ceiling) or
# top-50 (floor) player at the position (in the career view: ever, within the 10 seasons). No single time window to choose.
WINDOWS = {"five": (1, 2, 3, 5), "career": (queries.CAREER,)}
MODES = {
    "current": {"order": "value_now", "columns": [("Value", "value_now")]},
    "ceiling": {"order": "peak_ceiling", "columns": [("Ceiling", "peak_ceiling"), ("Top-10 %", "best_top10")]},
    "floor": {"order": "peak_floor", "columns": [("Floor", "peak_floor"), ("Top-50 %", "best_top50")]},
}


def add_projection(pool: pd.DataFrame, window: str = "five") -> pd.DataFrame:
    """`peak_ceiling`, `peak_floor` (best over the horizons), `best_top10`, `best_top50` (best chance over the horizons, in %)."""
    out = pool.copy()
    for target, source, scale in (
        ("peak_ceiling", "ceiling", 1), ("peak_floor", "floor", 1), ("best_top10", "p_top10", 100), ("best_top50", "p_top50", 100)
    ):  # fmt: skip
        columns = [f"{source}_h{h}" for h in WINDOWS[window] if f"{source}_h{h}" in out.columns]
        if columns:
            out[target] = out[columns].max(axis=1) * scale
    return out


def _players(
    rows: pd.DataFrame, mode: str, with_slot: bool, slot: str | None = None, with_org: bool = False
) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "ID": rows["ID"],
            "Player": rows["First Name"].astype(str).str[0] + ". " + rows["Last Name"].astype(str),
            "Age": rows["Age"],
            "Lev": rows["level_name"],
        }
    )
    if with_org:  # the organization of each player, when several are listed together
        out.insert(2, "Org", rows["org"])
    if with_slot:
        out.insert(1, "Pos", rows["slot"])
    if mode == "current" and with_slot:
        for label, column in (("Off", "bat_value"), ("Def", "field_value"), ("Run", "run_value")):
            if column in rows:  # run values above the league's average player at the position
                out[label] = rows[column]
    for label, column in MODES[mode]["columns"]:
        out[label] = rows[column]
    out["Slot"] = rows["slot"] if with_slot else slot  # the position the ratings card is for
    return out.reset_index(drop=True)


def _best(pool: pd.DataFrame, slots: list[str], by_player: bool, order: str) -> pd.DataFrame:
    """Every player of the pool at the slots, best first (once each, at his best slot, when by_player)."""
    rows = pool[pool["slot"].isin(slots)]
    if rows.empty:
        return rows
    ranked = rows.sort_values(order, ascending=False, kind="stable")
    return ranked.drop_duplicates("ID") if by_player else ranked


def org_overview(
    frame: pd.DataFrame, org: str, mode: str = "current", minors_only: bool = False, window: str = "five"
) -> dict[str, pd.DataFrame]:
    """The tables of one organization (see pool_overview)."""
    return pool_overview(frame, "org", org, mode, minors_only, window)


def pool_overview(
    frame: pd.DataFrame, pool_name: str, org: str | None = None, mode: str = "current", minors_only: bool = False,
    window: str = "five", with_org: bool = False, positions: list[str] | None = None,
) -> dict[str, pd.DataFrame]:  # fmt: skip
    """Tables keyed by title: 'Position players', one per hitting position, 'Starting pitchers', 'Relief pitchers'.

    Position players appear once, at their best position. The three long tables hold every player of the organization, best
    first (the screen shows the first TOP_* rows and sorts the whole table when a header is clicked); each position table
    holds its first TOP_PER_POSITION players, everybody who qualifies there. `pool_name` is a pool of queries.select_pool: an
    organization ("org"), the draft class ("draft") or the IAFA prospects ("iafa"). `minors_only` leaves out the MLB roster;
    `window` ("five" or "career") picks the horizons of the floor / ceiling modes; `with_org` adds an Org column; `positions`
    limits the Position players table to those positions: a player is then listed at the best of them, with his value there.
    """
    pool = queries.select_pool(frame, pool_name, org)
    if minors_only:
        pool = pool[pool["level_name"] != "MLB"]
    if mode != "current":
        pool = add_projection(pool, window)
    order = MODES[mode]["order"]
    tables = {
        "Position players": _players(
            _best(pool, positions or HITTER_SLOTS, True, order), mode, with_slot=True, with_org=with_org
        )
    }
    for slot in HITTER_SLOTS:
        rows = _best(pool, [slot], False, order).head(TOP_PER_POSITION)
        tables[slot] = _players(rows, mode, with_slot=False, slot=slot, with_org=with_org)
    tables["Starting pitchers"] = _players(
        _best(pool, ["SP"], False, order), mode, with_slot=False, slot="SP", with_org=with_org
    )
    tables["Relief pitchers"] = _players(
        _best(pool, ["RP"], False, order), mode, with_slot=False, slot="RP", with_org=with_org
    )
    return tables


def organizations(frame: pd.DataFrame) -> list[str]:
    return sorted(o for o in frame["org"].dropna().unique() if o not in ("FA", "nan", ""))

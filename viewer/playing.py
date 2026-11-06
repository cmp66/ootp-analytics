"""Who plays: the value of a player in a slot of a level, in parts. Pure functions, no UI.

The planner of the level moves page decides where a player is and, with this, what he does there. The value of a place (a slot of
a level) for a player has three parts, in one scale (utility, like the numbers of `viewer/roster.py`):

* **development**: his ceiling (its standing among the league's minor leaguers of his kind) x the years of development window he
  has left (at most `window_cap`) x how young he is for the level (a year younger than the median age of the level counts
  `age_step` more) x the share of the season he plays at the slot (a starter 1, the bench `bench_share`) x a **momentum**
  multiplier from his recent growth: rating and potential gained over the last year carry on (in Model-Sim, with talent change
  randomness on, +1 rating point is +0.8 over three years for a hitter, +0.38 for a pitcher; the potential adds 0.19 / 0.03);
* **learning**: at a position that is not his best he builds the rating there, `learn_by_level` points per 100 innings (measured
  in
  the weekly runs of Development-Analysis: the rating at a position grows with innings at it and not otherwise, and the lower the
  level the faster: 3.3 points per 100 innings in the rookie leagues, 1.5 in A ball, 0.6 at AAA / AA, 0.08 in the majors), at most
  up to his potential at the position (set by his range, arm and the other ratings: a good shortstop can play every infield
  position, others have limits); without a rating he starts at `start_rating`, and without a potential there is no learning;
* **primary first**: the user's practice is to train a player at his own position before moving him around, especially at C, SS
  and CF, where the rating at the position must be 70 or higher in the majors and the climb takes time. So the learning at his
  primary position is worth more (`primary_premium`, `key_premium` for `key_positions`), and learning at another position counts
  only once the primary is close to its potential (`primary_ready`, `key_ready`; before that `before_ready` of it);
* **only for a player with a future**: the learning counts by his ceiling (its standing among the league's minor leaguers of his
  kind) and not at all once his development window is over;
* **unfamiliar position**: a player learning a position pays in performance (the game models his hitting suffering while he
  deals with it, less as he learns): `unfamiliar_hit` x the development value x how far he is from his potential there is
  subtracted at a position that is not his primary. A prospect in the window in which his growth matters most is better at his
  best position;
* **current**: how he stands among the league's players of his kind now, at the slot (a floor: a level has to field a team).

`Playing` holds the settings (`[playing]` in config/minors.toml; hand tuned weights are guesses, the carry-overs and the learning
are measured). The input table has, per player: ID, Age, pitcher, Window, ceiling_pct, optionally d_cur and d_pot (the change of
the key ratings and of the potential over the last year, NaN when there is no history), the primary position `slot`, and per slot
S the standing `std_S` (0 to 1, NaN when he cannot play it), `rating_S` and `pot_S` (his position rating and potential).
"""

import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

from viewer.minors import MINORS_CONFIG

FIELD = ("C", "1B", "2B", "3B", "SS", "LF", "CF", "RF")
HITTER_SLOTS = (*FIELD, "DH", "BENCH")
PITCHER_SLOTS = ("SP", "RP")
SLOTS = (*HITTER_SLOTS, *PITCHER_SLOTS)


@dataclass(frozen=True)
class Playing:
    development: float = 10.0
    learning: float = 0.5
    current: float = 6.0
    starter_share: float = 1.0
    bench_share: float = 0.15
    bullpen_share: float = 0.30
    age_step: float = 0.15
    age_min: float = 0.5
    age_max: float = 1.5
    window_cap: int = 5
    innings: float = 1000.0
    momentum_scale: float = 0.10
    momentum_min: float = 0.5
    momentum_max: float = 2.0
    carry_hitters: float = 0.80
    carry_hitters_potential: float = 0.19
    carry_pitchers: float = 0.38
    carry_pitchers_potential: float = 0.03
    start_rating: float = 25.0
    joint_weight: float = 0.5
    primary_premium: float = 1.5
    key_premium: float = 2.0
    key_positions: tuple = ("C", "SS", "CF")
    primary_ready: float = 0.80
    key_ready: float = 0.90
    before_ready: float = 0.20
    unfamiliar_hit: float = 0.30
    learn_by_level: dict = field(default_factory=lambda: {"mlb": 0.08, "upper": 0.58, "a": 1.51, "rookie": 3.29})


def load_playing(path: Path = MINORS_CONFIG) -> Playing:
    raw = tomllib.loads(path.read_text()) if path.exists() else {}
    values = dict(raw.get("playing", {}))
    learn = values.pop("learn_by_level", None)
    if "key_positions" in values:
        values["key_positions"] = tuple(values["key_positions"])
    unknown = set(values) - set(Playing.__dataclass_fields__)
    if unknown:
        raise ValueError(f"{path}: unknown settings in [playing]: {sorted(unknown)}")
    settings = replace(Playing(), **values)
    return (
        replace(settings, learn_by_level={**settings.learn_by_level, **{k: float(v) for k, v in learn.items()}})
        if learn
        else settings
    )


def share(slot: str, p: Playing) -> float:
    """The share of a season's playing time at a slot: a lineup slot and a rotation spot are full time."""
    if slot == "BENCH":
        return p.bench_share
    if slot == "RP":
        return p.bullpen_share
    return p.starter_share


def age_factor(age, median_age: float, p: Playing):
    """How much more a player counts for being younger than the median of the level (less for being older)."""
    return np.clip(1.0 + p.age_step * (median_age - np.asarray(age, dtype=float)), p.age_min, p.age_max)


def expected_growth(d_cur, d_pot, pitcher, p: Playing):
    """The extra rating points expected over the next three years from last year's change of rating and of potential (0 when there
    is no history)."""
    d_cur, d_pot = np.nan_to_num(np.asarray(d_cur, dtype=float)), np.nan_to_num(np.asarray(d_pot, dtype=float))
    pitcher = np.asarray(pitcher, dtype=bool)
    return np.where(
        pitcher,
        p.carry_pitchers * d_cur + p.carry_pitchers_potential * d_pot,
        p.carry_hitters * d_cur + p.carry_hitters_potential * d_pot,
    )


def momentum(points, p: Playing):
    """The multiplier of the development value from the expected extra growth (1 without history)."""
    return np.clip(1.0 + p.momentum_scale * np.asarray(points, dtype=float), p.momentum_min, p.momentum_max)


def development(ceiling_pct, window, age, median_age: float, slot: str, points, p: Playing):
    """The development part of a place: ceiling x window x age for the level x share of the season x momentum, x weight."""
    years = np.minimum(np.asarray(window, dtype=float), p.window_cap) / p.window_cap
    ceiling = np.nan_to_num(np.asarray(ceiling_pct, dtype=float))
    return p.development * ceiling * years * age_factor(age, median_age, p) * share(slot, p) * momentum(points, p)


def level_group(depth: int) -> str:
    """The group of a level for the learning rate, by its place on the ladder (0 = the majors, 1 AAA, 2 AA, 3 and 4 the two A
    levels, below them the rookie levels and the international complex)."""
    return "mlb" if depth <= 0 else "upper" if depth <= 2 else "a" if depth <= 4 else "rookie"


def learning(slot: str, rating, potential, depth: int, p: Playing):
    """The rating points a player gains at a position by playing it for a season at a level `depth` places down the ladder, at
    most up to his potential there; 0 away from the field positions and without a potential."""
    if slot not in FIELD:
        return np.zeros(len(np.atleast_1d(rating)))
    rating = np.nan_to_num(np.asarray(rating, dtype=float), nan=p.start_rating)
    potential = np.asarray(potential, dtype=float)
    room = np.where(np.isnan(potential), 0.0, np.maximum(potential - rating, 0))
    return np.minimum(p.learn_by_level[level_group(depth)] * p.innings * share(slot, p) / 100.0, room)


def familiarity(rating, potential, p: Playing):
    """How far he is toward his potential at a position (0 to 1): 0 at the 20 of an unknown position, 1 at the potential.
    Without a potential there is no ceiling to measure against: 0."""
    rating = np.nan_to_num(np.asarray(rating, dtype=float), nan=p.start_rating)
    potential = np.asarray(potential, dtype=float)
    span = potential - 20.0
    return np.where(
        np.isnan(potential) | (span <= 0), 0.0, np.clip((rating - 20.0) / np.where(span > 0, span, 1.0), 0.0, 1.0)
    )


def ready_level(position: str, p: Playing) -> float:
    """How close to his potential a player's primary position must be before learning another counts in full."""
    return p.key_ready if position in p.key_positions else p.primary_ready


def slot_utilities(
    players: pd.DataFrame, median_age: float, depth: int = 1, p: Playing | None = None, slots=SLOTS
) -> pd.DataFrame:
    """The value of every player in every slot he can play at a level `depth` places down the ladder whose median age is
    `median_age`, in parts.

    Returns ID, slot, development, learning, hit (what an unfamiliar position costs), current, total, gain (rating points
    learned), ready (whether his primary position is trained enough for another to count in full) and a reason. A slot he cannot
    play (`std_<slot>` is NaN) is left out; BENCH is open to every hitter, SP and RP to pitchers (with a standing there).
    """
    p = p or Playing()
    nan = pd.Series(np.nan, index=players.index)
    pitcher = players["pitcher"].astype(bool).to_numpy()
    points = expected_growth(players.get("d_cur", nan), players.get("d_pot", nan), pitcher, p)
    primary_f = np.zeros(len(players))  # how far each hitter's own position is trained
    ready = np.ones(len(players), dtype=bool)
    for position in FIELD:
        mine = (players["slot"] == position).to_numpy()
        if mine.any():
            f = familiarity(players.get(f"rating_{position}", nan)[mine], players.get(f"pot_{position}", nan)[mine], p)
            primary_f[mine] = f
            ready[mine] = f >= ready_level(position, p)
    rows = []
    for slot in slots:
        if slot in PITCHER_SLOTS:
            ok = pitcher & players.get(f"std_{slot}", nan).notna().to_numpy()
        elif slot == "BENCH":
            ok = ~pitcher
        else:
            ok = ~pitcher & players.get(f"std_{slot}", nan).notna().to_numpy()
        if not ok.any():
            continue
        sub = players[ok]
        dev = development(sub["ceiling_pct"], sub["Window"], sub["Age"], median_age, slot, points[ok], p)
        mine = (sub["slot"] == slot).to_numpy()
        gain = np.zeros(len(sub))
        hit = np.zeros(len(sub))
        gate = np.ones(len(sub))
        if slot in FIELD:
            rating = sub.get(f"rating_{slot}", pd.Series(np.nan, index=sub.index))
            potential = sub.get(f"pot_{slot}", pd.Series(np.nan, index=sub.index))
            gain = learning(slot, rating, potential, depth, p)
            premium = p.key_premium if slot in p.key_positions else p.primary_premium
            gate = np.where(mine, premium, np.where(ready[ok], 1.0, p.before_ready))
            # only a player with a future is worth the time: the learning counts by his ceiling, and not at all past his window
            gate = (
                gate
                * np.nan_to_num(sub["ceiling_pct"].to_numpy(dtype=float))
                * (sub["Window"].to_numpy(dtype=float) > 0)
            )
            hit = np.where(mine, 0.0, p.unfamiliar_hit * dev * (1.0 - familiarity(rating, potential, p)))
        standing = sub.get(f"std_{slot}", sub.get("mean", pd.Series(0.5, index=sub.index))).fillna(0.0).to_numpy()
        part = pd.DataFrame(
            {"ID": sub["ID"].to_numpy(), "slot": slot, "development": dev, "learning": p.learning * gain * gate,
             "hit": hit, "current": p.current * standing * share(slot, p), "gain": gain, "ready": ready[ok],
             "primary": mine}
        )  # fmt: skip
        rows.append(part)
    columns = ["ID", "slot", "development", "learning", "hit", "current", "gain", "ready", "primary", "total", "reason"]
    if not rows:
        return pd.DataFrame(columns=columns)
    out = pd.concat(rows, ignore_index=True)
    out["total"] = out["development"] + out["learning"] - out["hit"] + out["current"]
    out["reason"] = [reason(r) for r in out.itertuples()]
    return out[columns]


def reason(r) -> str:
    """The words for a place: the position learning and what it costs, then development and current value."""
    text = f"development {r.development:.1f}, current {r.current:.1f}"
    if r.gain > 0 and r.primary:
        return f"{r.slot} (his position): trains his {r.slot} rating +{r.gain:.1f}; {text}"
    if r.gain > 0:
        wait = "" if r.ready else "; his own position is not trained enough yet, so this counts little"
        return (
            f"plays {r.slot}: builds his {r.slot} rating +{r.gain:.1f}, costs {r.hit:.1f} in performance{wait}; {text}"
        )
    return f"{r.slot}: {text}"

"""Park adjustment of the values of the viewer: what a player is worth if half his games are played in a given park.

The values of the outlook files are in a neutral park (the value models are trained on park-adjusted outcomes). For a hitter the
park moves his home runs, doubles, triples and hits by their park factors (the left or right factor by the side he bats from;
singles are what is left of the hits), valued with the run weights of the league: `home_share` of his games (config/saves.toml)
are played in the park, the rest in neutral parks, exactly as stats/batting.py adjusts the statistics. His neutral-park rates per
plate appearance come from his ratings (model/park.py, scripts/predict_park.py). For a pitcher the park scales the runs he allows
(its run factor): a pitcher who allows R runs per 200 innings in a neutral park allows R times the effective factor there.
Fielding and baserunning are not affected. Everything that is a value in runs is moved: the value now and its parts and the
median, floor, ceiling and quartiles of the projections; the chances of being a top-10 or top-50 player are not.
"""

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from model.park import SWITCH
from stats import parks
from util import paths
from util.config import get_save
from viewer import queries

VALUE_COLUMNS = re.compile(
    r"^(value_now|bat_value|pit_value|(value|q25|q75|floor|ceiling|rank_score|season_value|season_floor|season_ceiling)_h\d+)$"
)
PITCHER_SLOTS = ("SP", "RP")
DEFAULT_RA9 = 4.5  # runs per nine innings of a league without team pitching totals
RATES = ("HR", "DOUBLE", "TRIPLE", "H")


@dataclass
class ParkContext:
    factors: pd.DataFrame  # team code -> park factors (stats.parks.FACTOR_COLUMNS renamed)
    home_share: float
    coef: dict  # the wOBA weights of the league (1B, 2B, 3B, HR) and `scale` (wOBA_SCALE)
    ra9: float  # league runs per nine innings
    rates: pd.DataFrame  # ID -> left_share, HR, DOUBLE, TRIPLE, H (neutral-park rates per PA)

    @property
    def runs_per_200_ip(self) -> float:
        return self.ra9 * 200 / 9

    def teams(self) -> list[str]:
        return sorted(self.factors.index)


def _innings(value) -> float:
    whole = np.floor(value)
    return whole + (value - whole) * 10 / 3  # 145.1 is 145 and a third


def league_ra9(save: str, season: int) -> float:
    path = paths.output_file(save, season, "team-pitching")
    if not path.exists():
        return DEFAULT_RA9
    table = pd.read_csv(path)
    if not {"IP", "R"} <= set(table.columns) or table["IP"].isna().all():
        return DEFAULT_RA9
    innings = _innings(pd.to_numeric(table["IP"], errors="coerce")).sum()
    return float(table["R"].sum() / innings * 9) if innings > 0 else DEFAULT_RA9


def load_context(save: str, season: int) -> ParkContext | None:
    """The park factors by team, the league's run weights and the players' neutral-park rates; None when the save has no ballpark
    file or no rates (train the park models and run scripts/update_all.py)."""
    config = get_save(save)
    rates_path = paths.output_file(save, season, "park-rates")
    woba_path = paths.output_file(save, season, "woba-calcs")
    if config.source != "export" or not config.team_names or not rates_path.exists() or not woba_path.exists():
        return None
    try:
        file = parks.ballpark_file(config, season)
    except FileNotFoundError:
        return None
    if file is None:
        return None
    table = parks.load_ballparks(file)
    codes = {name: code for code, name in config.team_names.items()}
    factors = table.rename(index=codes)
    factors = factors[factors.index.isin(config.team_names)]
    weights = pd.read_csv(woba_path).set_index("Stat")["Value"]
    coef = {name: float(weights[f"coef_{name}"]) for name in ("1B", "2B", "3B", "HR")} | {
        "scale": float(weights["wOBA_SCALE"])
    }
    rates = pd.read_csv(rates_path).set_index("ID")
    return ParkContext(factors, config.home_share, coef, league_ra9(save, season), rates)


def effective(ctx: ParkContext, teams: pd.Series) -> pd.DataFrame:
    """The factors a player faces when `home_share` of his games are in the park of his team: one row per entry of `teams` (a team
    code), neutral (1) for a team without a park."""
    table = ctx.factors.reindex(teams.to_numpy())[parks.BATTING_FACTORS + parks.PITCHING_FACTORS]
    out = ctx.home_share * table + (1 - ctx.home_share)
    return out.fillna(1.0).set_axis(teams.index)


def side_left_share(bats: pd.Series, side: str) -> pd.Series:
    """The share of a hitter's plate appearances from the left side against pitchers who throw with `side` ("right" or "left"):
    left-handed hitters 1, right-handed 0, a switch hitter bats left against a right-hander and right against a left-hander.
    NaN for a hitter whose batting hand is not known."""
    codes = pd.to_numeric(bats, errors="coerce")
    share = pd.Series(np.nan, index=bats.index)
    share[codes == 1] = 0.0
    share[codes == 2] = 1.0
    share[codes == SWITCH] = 1.0 if side == "right" else 0.0
    return share


def hitter_shift(ctx: ParkContext, ids: pd.Series, eff: pd.DataFrame, left_share: pd.Series | None = None) -> pd.Series:
    """The runs per 600 PA a hitter gains (or loses) in the park: his neutral-park rates times what each event changes by, with
    the run weights of the league. Zero for a player without rates. `left_share` (same index as `ids`) replaces his usual share of
    plate appearances from the left side, for the value against one kind of pitcher; NaN keeps the usual one."""
    rates = ctx.rates.reindex(ids.to_numpy()).set_axis(ids.index)
    left = rates["left_share"].fillna(0.5)
    if left_share is not None:
        left = left_share.fillna(left)
    f_avg = left * eff["AVG_L"] + (1 - left) * eff["AVG_R"]
    f_hr = left * eff["HR_L"] + (1 - left) * eff["HR_R"]
    hits, doubles, triples, homers = rates["H"], rates["DOUBLE"], rates["TRIPLE"], rates["HR"]
    neutral_singles = (hits - doubles - triples - homers).clip(lower=0)
    singles = (hits * f_avg - doubles * eff["D"] - triples * eff["T"] - homers * f_hr).clip(lower=0)
    runs = (
        (
            ctx.coef["1B"] * (singles - neutral_singles)
            + ctx.coef["2B"] * doubles * (eff["D"] - 1)
            + ctx.coef["3B"] * triples * (eff["T"] - 1)
            + ctx.coef["HR"] * homers * (f_hr - 1)
        )
        * 600
        / ctx.coef["scale"]
    )
    return runs.fillna(0.0)


def adjust(frame: pd.DataFrame, ctx: ParkContext, team: str | None = None, own: bool = False) -> pd.DataFrame:
    """A copy of the outlook frame with the values in runs moved to the park of `team` (every row), or with `own` to the park of
    each row's own organization (a player without one, a free agent, is left alone). Adds `park_effect`, the change of the value
    now.
    """
    out = frame.copy()
    teams = out["org"].astype(str) if own else pd.Series(str(team), index=out.index)
    eff = effective(ctx, teams)
    pitcher = out["slot"].isin(PITCHER_SLOTS)
    shift = hitter_shift(ctx, out["ID"], eff).where(~pitcher, 0.0)
    factor = eff["RUNS"].where(pitcher, 1.0)  # a pitcher's runs allowed scale with the run factor of the park
    base = ctx.runs_per_200_ip
    for column in [c for c in out.columns if VALUE_COLUMNS.match(c)]:
        if column == "bat_value":
            out[column] = out[column] + shift
        elif column == "pit_value":
            out[column] = factor * out[column] - (factor - 1) * base
        else:
            out[column] = np.where(pitcher, factor * out[column] - (factor - 1) * base, out[column] + shift)
    for (
        side
    ) in queries.SIDES:  # the value against one kind of pitcher: a switch hitter bats from the other side of the plate
        column = f"value_vs_{side}"
        if column in out:
            bats = out["B"] if "B" in out else pd.Series(np.nan, index=out.index)
            side_shift = hitter_shift(ctx, out["ID"], eff, side_left_share(bats, side))
            out[column] = out[column] + side_shift.where(~pitcher, 0.0)
    out["park_effect"] = out["value_now"] - frame["value_now"]
    return out

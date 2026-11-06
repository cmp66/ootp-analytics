"""Scoring with the outlook models (plan 16, stage 4): the columns for every player of a season, for a dump or an export save.

The path is the one of the training tables, built from the player-data of one season alone: the plan 14 models give the
league-relative value of every player at every slot, the gates and the ranks decide who is ranked where, and the features
of the outlook models are built by the same code as in training (model/outlook_models.family_features). Two frames come out:

- slots: one row per player and ranked slot: the value now, its rank and tier, and for each horizon the median, floor and
  ceiling of the value (the 80 % band, widened by the calibration shift stored with the models), the chance of being top 10 /
  top 50 at the slot, a `rank_score` that blends floor, median and ceiling, and the value per season once the availability is
  taken into account (median, floor, ceiling) with the injury cost.
- players: one row per player: the chance of staying in the league, availability (10th / 50th / 90th percentile) with the
  availability of a normal-proneness player of the same profile, their ratio and the injury risk, and a depth flag.

Values are in the units of the plan 14 models (runs per season at full playing time); all of them are relative to the league
that is scored (model/relative.py).
"""

import numpy as np
import pandas as pd

from model import artifact, formats, outlook_data, outlook_models, registry, scales
from model.outlook_models import (
    AVAIL_FEATURES,
    CAREER,
    FAMILIES,
    HORIZONS,
    KNOWN_HORIZONS,
    PLAYER_FEATURES,
    VALUE_QUANTILES,
    feature_names,
)
from model.quantiles import QUANTILE_LEVELS, exceed_probability  # noqa: F401
from util.config import get_save

RANK_WEIGHTS = (0.25, 0.5, 0.25)  # floor, median, ceiling
DEPTH_PERCENTILE = 0.25  # a player whose best rating value is below this share of the league is depth, not a prospect
VALUE_GRID = np.linspace(0.05, 0.95, 19)
AVAIL_GRID = np.linspace(0.1, 0.9, 9)
RATIO_CAP = 1.3  # the most a durable player is credited with against a normal one


# ---------------------------------------------------------------- loading


class OutlookModels:
    """The stage 3 artifacts of one save, by kind, family and horizon."""

    def __init__(self, save: str, ratings: str):
        folder = artifact.models_dir(save, ratings) / outlook_models.OUTLOOK_DIR
        if not folder.exists():
            raise FileNotFoundError(f"{folder} not found; run scripts/train_outlook_models.py {save}")
        self.items = {
            p.stem: artifact.load_artifact(save, ratings, f"{outlook_models.OUTLOOK_DIR}/{p.stem}")
            for p in folder.glob("*.joblib")
        }
        self.save, self.ratings = save, ratings

    def horizons(self) -> tuple[int, ...]:
        """The horizons every family has value models for. Models trained on other features than the current ones (a folder
        keeps the files of horizons that were dropped from training, such as 2 and 3 seasons ahead, with the whole age where
        the models now read the fine age) are not used."""
        found = {family: set() for family in FAMILIES}
        for name, item in self.items.items():
            parts = name.split("-")
            if parts[0] == "value" and parts[1] in found and parts[2].startswith("h"):
                if set(item.features) <= set(outlook_models.feature_names(parts[1])):
                    found[parts[1]].add(int(parts[2][1:]))
        common = set.intersection(*found.values()) if found else set()
        return tuple(sorted(common))

    def has(self, name: str) -> bool:
        return name.split("/")[-1] in self.items

    def get(self, name: str):
        name = name.split("/")[-1]
        if name not in self.items:
            raise KeyError(
                f"outlook model {name} not found for {self.save}; retrain with scripts/train_outlook_models.py"
            )
        return self.items[name]

    def quantiles(self, kind: str, family: str, horizon: int, X: pd.DataFrame, levels) -> tuple[dict, float]:
        """Sorted predictions of the quantile models and the band shift stored with them."""
        predictions = {}
        shift = 0.0
        for q in levels:
            model = self.get(outlook_models.artifact_name(kind, family, horizon, f"q{int(q * 100)}"))
            predictions[q] = model.predict(X)
            shift = model.notes.get("band_shift", shift)
        stacked = np.sort(np.column_stack([predictions[q] for q in levels]), axis=1)
        return {q: stacked[:, i] for i, q in enumerate(levels)}, float(shift)

    def probability(
        self,
        kind: str,
        family: str,
        horizon: int,
        X: pd.DataFrame,
        slots: pd.Series | None = None,
        teams: int | None = None,
    ) -> np.ndarray:
        """Calibrated chance of a tier; with `slots` (the slot name of each row) and `teams` it is also rescaled to the league."""
        model = self.get(outlook_models.artifact_name(kind, family, horizon))
        p = model.predict(X)
        calibration = model.notes.get("calibration")
        p = outlook_models.apply_probability_calibration(p, calibration) if calibration else p
        expected = model.notes.get("expected_count")
        if slots is None or not expected:
            return p
        return normalize_to_league(p, slots, expected, kind, teams, model.notes.get("teams"))


# ---------------------------------------------------------------- the inputs of one season


def season_inputs(save: str, season: int, models_from: str | None = None) -> dict:
    """State, ranked slot values and the player table of one season, from its player-data alone."""
    config = get_save(save)
    models_from = models_from or save
    plan14 = outlook_data._load_artifacts(models_from, scales.model_set(config.ratings, config.scales))
    state = outlook_data.season_state(save, season)
    if config.scales:  # the position minimums and the outlook models know the 20-80 scale: give them the expected value
        conversion = scales.Conversion.load(scales.version_of(config.ratings), relative=config.relative_ratings)
        state = conversion.expect_frame(state, scales.profile_columns(save, season, conversion))
    scored, _ = registry.score_season(save, season, plan14)
    values = outlook_data.slot_values(state, scored)
    gates = outlook_data.load_gates(config.ratings)
    if gates is None:
        raise FileNotFoundError(
            f"no position minimums for {config.ratings}; run scripts/build_outlook_tables.py on a dump save of it"
        )
    kept = []
    for slot, group in values.groupby("slot"):
        ok = group["ID"].map(outlook_data.qualifies(state, slot, gates)).fillna(False).astype(bool)
        kept.append(group[ok.to_numpy()])
    qualified = pd.concat(kept, ignore_index=True)
    members = state.index[
        outlook_data.in_organization(state)
    ]  # free agents and amateurs are placed on the organizations' scale
    reference = members if config.source == "export" else None
    ranked = outlook_data.rank_slots(qualified, outlook_data.team_count(save, season), reference).assign(season=season)
    bounds = outlook_data.load_trait_bounds(config.ratings) if config.source == "dump" else {}
    if config.source == "dump" and bounds is None:
        raise FileNotFoundError(f"no trait cut points for {config.ratings}; run scripts/build_outlook_tables.py")
    state = outlook_data.with_trait_categories(state, bounds or {})
    people = state.reset_index()
    people["LPOS"] = people["LPOS"].astype(str)
    return {"state": people, "ranked": ranked, "season": season, "teams": outlook_data.team_count(save, season)}


def scoring_player_table(people: pd.DataFrame, ranked: pd.DataFrame) -> pd.DataFrame:
    """Player-level feature table (best value, depth) in the layout of outlook_models.player_table."""
    return outlook_models.player_table(people, ranked)


def normalize_to_league(
    p: np.ndarray, slots: pd.Series, expected: dict, tier: str, teams: int | None, train_teams: int | None
) -> np.ndarray:
    """Rescale the tier chances so that, slot by slot, they add up to the expected number of players (see
    outlook_models.expected_counts), scaled by the tier cutoff of this league against the training league.

    The classifiers learned the value scale of the training league; in a league with a weaker or stronger top end the
    ranking of the players is right but the level of the chances is not. The rescaling keeps the order within a slot and
    caps every chance at 1 (re-spreading what the cap removes over the others)."""
    cutoffs = outlook_data.tier_cutoffs
    ratio = 1.0
    if teams and train_teams:
        ratio = cutoffs(teams)[tier] / cutoffs(train_teams)[tier]
    out = np.asarray(p, dtype=float).copy()
    names = slots.to_numpy()
    for slot, target in expected.items():
        mask = names == slot
        wanted = min(target * ratio, float((out[mask] > 0).sum()))  # at most one per player
        if not mask.any() or out[mask].sum() <= 0 or wanted <= 0:
            continue
        values = out[mask]
        low, high = 0.0, 1.0 / values[values > 0].min()  # the scale at which every chance would be 1
        for _ in range(60):  # bisection on the scale c: sum(min(1, c p)) grows with c
            middle = (low + high) / 2
            if np.minimum(values * middle, 1.0).sum() < wanted:
                low = middle
            else:
                high = middle
        out[mask] = np.minimum(values * high, 1.0)
    return out


TIER_BLEND = (
    0.5  # weight of the classifier in the tier chance; the rest comes from the value distribution and the thresholds
)


def tier_thresholds(ranked: pd.DataFrame, teams: int) -> dict:
    """The value now of the player at rank 10 / 50 (scaled by the number of teams) of every slot of the league being scored."""
    cutoffs = outlook_data.tier_cutoffs(teams)
    out = {tier: {} for tier in ("top10", "top50")}
    for slot, group in ranked.groupby("slot"):
        values = np.sort(group["value"].to_numpy())[::-1]
        for tier in out:
            out[tier][slot] = float(values[min(cutoffs[tier], len(values)) - 1]) if len(values) else np.inf
    return out


# ---------------------------------------------------------------- distributions


def quantile_function(levels: np.ndarray, known_levels, known_values: np.ndarray) -> np.ndarray:
    """Values at `levels` for each row from its quantiles at `known_levels`, linear between them and along the last segment
    outside them. known_values is (rows, k) and sorted along k."""
    known_levels = np.asarray(known_levels, dtype=float)
    out = np.empty((known_values.shape[0], len(levels)))
    for j, level in enumerate(levels):
        segment = int(np.clip(np.searchsorted(known_levels, level) - 1, 0, len(known_levels) - 2))
        low, high = known_levels[segment], known_levels[segment + 1]
        t = (level - low) / (high - low)
        out[:, j] = known_values[:, segment] + t * (known_values[:, segment + 1] - known_values[:, segment])
    return out


def season_value_distribution(value_q: np.ndarray, availability_q: np.ndarray, normal_median: np.ndarray) -> dict:
    """Value per season once availability is taken into account.

    value_q: (rows, 5) sorted quantiles of the value (10, 25, 50, 75, 90, with the band widening applied); availability_q:
    (rows, 3) quantiles (10, 50, 90) of the player's availability; normal_median: availability of the same player with normal
    proneness. The ratio own / normal scales a positive value (a below-average player who misses time costs nothing against
    the league average). The two spreads are combined as independent."""
    values = quantile_function(VALUE_GRID, VALUE_QUANTILES, value_q)
    avail = quantile_function(AVAIL_GRID, (0.1, 0.5, 0.9), availability_q)
    ratio = np.clip(avail / np.maximum(normal_median[:, None], 1e-6), 0.0, RATIO_CAP)
    combined = np.where(values[:, :, None] > 0, values[:, :, None] * ratio[:, None, :], values[:, :, None])
    flat = combined.reshape(len(values), -1)
    floor, median, ceiling = np.percentile(flat, [10, 50, 90], axis=1)
    return {"floor": floor, "median": median, "ceiling": ceiling}


# ---------------------------------------------------------------- scoring


def score_players(
    models: OutlookModels, ptable: pd.DataFrame, horizons=HORIZONS, depth_percentile: float = DEPTH_PERCENTILE
) -> pd.DataFrame:
    """Stay chance, availability, the availability of a normal-proneness twin, the injury risk and the depth flag."""
    out = ptable[["ID", "season", "Age", "LPOS", "level", "best_value", "best_pct"]].copy()
    out["depth_player"] = out["best_pct"] < depth_percentile
    normal = ptable[AVAIL_FEATURES].assign(Prone3_ord=1.0)
    for h in horizons:
        out[f"p_stay_h{h}"] = models.probability("stay", "player", h, ptable[PLAYER_FEATURES])
        if not models.has(
            outlook_models.artifact_name("avail", "player", h, "q50")
        ):  # a model set trained without playing time
            for column in ("avail_p10", "avail", "avail_p90", "avail_normal", "injury_risk"):
                out[f"{column}_h{h}"] = np.nan
            out[f"avail_ratio_h{h}"] = 1.0
            continue
        own, _ = models.quantiles("avail", "player", h, ptable[AVAIL_FEATURES], outlook_models.AVAIL_QUANTILES)
        twin, _ = models.quantiles("avail", "player", h, normal, outlook_models.AVAIL_QUANTILES)
        out[f"avail_p10_h{h}"], out[f"avail_h{h}"], out[f"avail_p90_h{h}"] = own[0.1], own[0.5], own[0.9]
        out[f"avail_normal_h{h}"] = twin[0.5]
        ratio = np.clip(own[0.5] / np.maximum(twin[0.5], 1e-6), 0.0, RATIO_CAP)
        out[f"avail_ratio_h{h}"] = ratio
        out[f"injury_risk_h{h}"] = 1.0 - ratio  # negative: more durable than a normal-proneness player
    return out


def score_slots(
    models: OutlookModels, inputs: dict, ptable: pd.DataFrame, players: pd.DataFrame, horizons=KNOWN_HORIZONS
) -> pd.DataFrame:
    """Floor / median / ceiling, tier chances and season value for every ranked slot of every player.

    The CAREER horizon (99) describes his best season within the next ten: its median / floor / ceiling are those of the peak
    value, its tier chances the chance of ever being top 10 / top 50 at the slot; it has no season value (no availability).
    """
    ranked, season = inputs["ranked"], inputs["season"]
    thresholds = tier_thresholds(ranked, inputs["teams"])
    first_real = min(
        h for h in horizons if h != CAREER
    )  # the career chance counts him only if he stays at least that long
    outputs = []
    for family, slots in FAMILIES.items():
        rows = ranked[ranked["slot"].isin(slots)][["ID", "season", "slot", "value", "pct", "rank", "tier"]]
        rows = rows.rename(columns={"value": "value_now", "pct": "pct_now", "rank": "rank_now", "tier": "tier_now"})
        if rows.empty:
            continue
        context = ptable[outlook_models.CONTEXT_COLUMNS]
        ratings = outlook_models.load_ratings(inputs["save"], [season], outlook_models.rating_columns_needed(family))
        frame = outlook_models.family_features(rows, context, ratings, family)
        X = frame[feature_names(family)]
        result = frame[["ID", "season", "slot_name", "value_now", "pct_now", "rank_now", "tier_now"]].rename(
            columns={"slot_name": "slot"}
        )
        per_player = players.set_index("ID")
        for h in horizons:
            q, shift = models.quantiles("value", family, h, X, VALUE_QUANTILES)
            now = frame["value_now"].to_numpy()
            low, high = q[0.1] - shift, q[0.9] + shift
            result[f"value_h{h}"] = now + q[0.5]
            result[f"q25_h{h}"], result[f"q75_h{h}"] = now + q[0.25], now + q[0.75]
            result[f"floor_h{h}"] = np.minimum(
                now + low, result[f"value_h{h}"]
            )  # a narrowed band never crosses the median
            result[f"ceiling_h{h}"] = np.maximum(now + high, result[f"value_h{h}"])
            names = frame["slot_name"]
            quantile_matrix = np.sort(
                np.column_stack([now + low, now + q[0.25], now + q[0.5], now + q[0.75], now + high]), axis=1
            )
            stay = per_player.reindex(frame["ID"])[f"p_stay_h{first_real if h == CAREER else h}"].to_numpy()
            for tier in ("top10", "top50"):
                from_classifier = models.probability(tier, family, h, X, names, inputs["teams"])
                limit = names.map(thresholds[tier]).to_numpy(dtype=float)
                from_values = stay * exceed_probability(quantile_matrix, limit)
                result[f"p_{tier}_h{h}"] = TIER_BLEND * from_classifier + (1 - TIER_BLEND) * from_values
            result[f"p_top10_h{h}"] = np.minimum(result[f"p_top10_h{h}"], result[f"p_top50_h{h}"])
            w_floor, w_median, w_ceiling = RANK_WEIGHTS
            result[f"rank_score_h{h}"] = (
                w_floor * result[f"floor_h{h}"] + w_median * result[f"value_h{h}"] + w_ceiling * result[f"ceiling_h{h}"]
            )
            if h == CAREER:
                continue
            joined = per_player.reindex(frame["ID"])
            avail = np.column_stack([joined[f"avail_p10_h{h}"], joined[f"avail_h{h}"], joined[f"avail_p90_h{h}"]])
            avail = np.sort(np.nan_to_num(avail, nan=0.6), axis=1)
            normal_median = np.nan_to_num(joined[f"avail_normal_h{h}"].to_numpy(), nan=0.6)
            value_q = np.sort(
                np.column_stack(
                    [now + q[0.1] - shift, now + q[0.25], now + q[0.5], now + q[0.75], now + q[0.9] + shift]
                ),
                axis=1,
            )
            season_value = season_value_distribution(value_q, avail, normal_median)
            result[f"season_value_h{h}"] = season_value["median"]
            result[f"season_floor_h{h}"], result[f"season_ceiling_h{h}"] = (
                season_value["floor"],
                season_value["ceiling"],
            )
            ratio = np.clip(avail[:, 1] / np.maximum(normal_median, 1e-6), 0.0, RATIO_CAP)
            result[f"injury_cost_h{h}"] = np.maximum(result[f"value_h{h}"].to_numpy(), 0.0) * (1.0 - ratio)
        outputs.append(result)
    return pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()


def outlook_for_season(
    save: str, season: int, models_from: str | None = None, horizons=None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(slots, players): the outlook columns of every player of a season, for the horizons the model set has."""
    config = get_save(save)
    models = OutlookModels(models_from or save, config.ratings)
    horizons = tuple(h for h in (horizons or KNOWN_HORIZONS) if h in models.horizons())
    if not horizons:
        raise FileNotFoundError(
            f"no outlook value models in {models.save}; run scripts/train_outlook_models.py {models.save}"
        )
    inputs = season_inputs(save, season, models_from)
    inputs["save"] = save
    ptable = scoring_player_table(inputs["state"], inputs["ranked"])
    players = score_players(models, ptable, tuple(h for h in horizons if h != CAREER))
    slots = score_slots(models, inputs, ptable, players, horizons)
    return slots, players


def level_number(level, ratings: str) -> float:
    return formats.export_level_number(level, ratings)

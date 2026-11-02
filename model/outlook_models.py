"""The outlook models (plan 16, stage 3): where a player's rating-based value is going, how likely he is to be a top 10 / top 50
player at a position, whether he stays in the league, and how much of a season he will be available.

Everything is trained on one game version's long CPU league from the tables of model/outlook_data.py, and uses only features
that both a dump and an in-game export hold (age, ratings and the game's potentials, work ethic / intelligence / proneness
tiers, level, height and weight).

Models, per horizon h (1, 2, 3, 5 seasons ahead):
- `value`: five quantiles (10, 25, 50, 75, 90) of the change of the slot value, for each family (hitters at C..RF and DH, pitchers
  at SP and RP), conditional on the player being ranked at the slot in t+h; the floor and ceiling are the 10th and 90th.
- `top10` / `top50`: classifiers for being ranked top 10 / top 50 at the same slot in t+h (a player who is gone counts as no).
- `stay`: classifier for being in the league in t+h.
- `avail`: quantiles (10, 50, 90) of the share of a full season he plays in t+h (time_share), a measure of availability.

Honest evaluation holds out the latest seasons with a gap of five seasons (the targets of the training rows reach five seasons
ahead) and compares with persistence baselines (value now, plus the mean change by age and slot).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression

from model import data, outlook_data
from model.artifact import Artifact, load_artifact, save_artifact

HORIZONS = (1, 2, 3, 5)
VALUE_QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)
AVAIL_QUANTILES = (0.1, 0.5, 0.9)
FAMILIES = {"hitters": outlook_data.HITTER_SLOTS, "pitchers": outlook_data.PITCHER_SLOTS}
SLOT_CODES = {slot: i for i, slot in enumerate([*outlook_data.HITTER_SLOTS, *outlook_data.PITCHER_SLOTS])}
GAP = 5  # seasons between the last training season and the first test season (the longest horizon)
TEST_SEASONS = 6

HITTER_RATINGS = [
    "BABIP", "GAP", "POW", "EYE", "K's", "BA P", "GAP P", "POW P", "EYE P", "K P",
    "SPE", "STE", "RUN", "B", "IF RNG", "IF ARM", "IF ERR", "TDP", "OF RNG", "OF ARM", "OF ERR", "C FRM", "C ABI", "C ARM",
]  # fmt: skip
PITCHER_RATINGS = [
    "STU", "MOV", "HRR", "PBABIP", "CON.1", "STU P", "MOV P", "HRR P", "PBABIP P", "CON.1 P", "STM", "HLD", "PIT", "VELO", "T",
]  # fmt: skip
SHARED_RATINGS = ["HT", "WT"]
TRAIT_FEATURES = ["WE_ord", "INT_ord", "Prone3_ord"]
RATING_COLUMNS = {"hitters": HITTER_RATINGS, "pitchers": PITCHER_RATINGS}
PLAYER_FEATURES = ["Age", "level", "best_value", "best_pct", "is_pitcher", *TRAIT_FEATURES, "HT", "WT"]
AVAIL_FEATURES = ["Age", "level", "best_value", "is_pitcher", *TRAIT_FEATURES]
OUTLOOK_DIR = "outlook"


def feature_names(family: str) -> list[str]:
    """Features of a family's value models and tier classifiers, in order."""
    slot_specific = ["slot_rating", "slot_pot"] if family == "hitters" else []
    return [
        "slot", "Age", "level", "value_now", "pct_now", "best_value", *slot_specific,
        *RATING_COLUMNS[family], *SHARED_RATINGS, *TRAIT_FEATURES,
    ]  # fmt: skip


# ---------------------------------------------------------------- tables


def load_ratings(save: str, seasons, columns: list[str]) -> pd.DataFrame:
    """The player-data ratings of the seasons as numbers, one row per player and season (columns missing there are NaN)."""
    frames = []
    for season in seasons:
        ratings = data.read_output(save, season, "player-data").drop_duplicates("ID")
        keep = [c for c in dict.fromkeys(columns) if c in ratings.columns]
        part = ratings[["ID", *keep]].copy()
        part[keep] = part[keep].apply(pd.to_numeric, errors="coerce")
        part["season"] = season
        frames.append(part)
    out = pd.concat(frames, ignore_index=True)
    for column in columns:
        if column not in out.columns:
            out[column] = np.nan
    return out


def player_table(players: pd.DataFrame, values: pd.DataFrame) -> pd.DataFrame:
    """One row per player and season for the player-level models (stay, availability) with the best value over the slots."""
    best = values.groupby(["season", "ID"]).agg(best_value=("value", "max"), best_pct=("pct", "max")).reset_index()
    table = players.merge(best, on=["season", "ID"], how="left")
    table["level"] = table["level"].fillna(0)
    table["is_pitcher"] = table["LPOS"].isin(outlook_data.PITCHER_LPOS).astype(float)
    floor_value, floor_pct = table["best_value"].quantile(0.001), 0.0
    table["best_value"] = table["best_value"].fillna(floor_value)  # not ranked anywhere: below the whole field
    table["best_pct"] = table["best_pct"].fillna(floor_pct)
    return table


def add_player_targets(table: pd.DataFrame, horizons=HORIZONS) -> pd.DataFrame:
    """stay_h (already in the players table) and avail_h: the time share in season t+h (NaN without stats or beyond the table)."""
    shares = table[["ID", "season", "time_share"]].rename(columns={"time_share": "avail"})
    out = table.copy()
    for h in horizons:
        later = shares.assign(season=shares["season"] - h).rename(columns={"avail": f"avail_h{h}"})
        out = out.merge(later, on=["ID", "season"], how="left")
        out[f"avail_h{h}"] = out[f"avail_h{h}"].where(out[f"observed_h{h}"].astype(bool))
    return out


def family_table(save: str, family: str, horizons=HORIZONS, tables: dict | None = None) -> pd.DataFrame:
    """Training / scoring rows of one family: one per player, season and slot he is ranked at, with features and targets.

    Targets per horizon h: `dvalue_h` (value at t+h minus value now, NaN if the player is gone or beyond the table),
    `top50_h`, `top10_h` (0 when gone or not ranked then, NaN beyond the table), `observed_h`."""
    tables = tables or outlook_data.read_tables(save)
    players, values, every = tables["players"], tables["values"], tables["slot_values"]
    if every is None:
        raise FileNotFoundError(f"{save}: no slot_values table; run scripts/build_outlook_tables.py {save}")
    slots = FAMILIES[family]
    rows = values[values["slot"].isin(slots)][["ID", "season", "slot", "value", "pct", "tier"]].rename(
        columns={"value": "value_now", "pct": "pct_now"}
    )
    ptable = player_table(players, values)
    context = ptable[
        [
            "ID",
            "season",
            "Age",
            "level",
            "best_value",
            "WE_ord",
            "INT_ord",
            "Prone3_ord",
            *[f"observed_h{h}" for h in horizons],
        ]
    ]
    frame = rows.merge(context, on=["ID", "season"], how="left")
    seasons = sorted(players["season"].unique())
    ratings = load_ratings(
        save,
        seasons,
        [*RATING_COLUMNS[family], *SHARED_RATINGS, *outlook_data.POSITION_RATINGS, *outlook_data.POTENTIAL_RATINGS],
    )
    frame = frame.merge(ratings, on=["ID", "season"], how="left")
    if family == "hitters":
        by_slot = {slot: i for i, slot in enumerate(slots)}
        position = frame[outlook_data.POSITION_RATINGS].to_numpy()
        potential = frame[outlook_data.POTENTIAL_RATINGS].to_numpy()
        index = frame["slot"].map(by_slot).to_numpy()
        valid = index < len(outlook_data.POSITION_RATINGS)  # DH has no position rating
        pick = np.where(valid, index, 0)
        frame["slot_rating"] = np.where(valid, position[np.arange(len(frame)), pick], np.nan)
        frame["slot_pot"] = np.where(valid, potential[np.arange(len(frame)), pick], np.nan)
    frame["slot"] = frame["slot"].map(SLOT_CODES).astype(float)
    slot_names = {code: name for name, code in SLOT_CODES.items()}
    keyed = frame.assign(slot_name=frame["slot"].map(slot_names))
    for h in horizons:
        later_value = every[["ID", "slot", "season", "value"]].assign(season=every["season"] - h)
        later_value = later_value.rename(columns={"slot": "slot_name", "value": f"value_h{h}"})
        keyed = keyed.merge(later_value, on=["ID", "slot_name", "season"], how="left")
        later_tier = values[["ID", "slot", "season", "tier"]].assign(season=values["season"] - h)
        later_tier = later_tier.rename(columns={"slot": "slot_name", "tier": f"tier_h{h}"})
        keyed = keyed.merge(later_tier, on=["ID", "slot_name", "season"], how="left")
        observed = keyed[f"observed_h{h}"].astype(bool)
        keyed[f"dvalue_h{h}"] = keyed[f"value_h{h}"] - keyed["value_now"]
        tier = keyed[f"tier_h{h}"]
        keyed[f"top50_h{h}"] = tier.isin(["top10", "top50"]).astype(float).where(observed)
        keyed[f"top10_h{h}"] = (tier == "top10").astype(float).where(observed)
    return keyed.drop(columns=["slot_name", *[f"tier_h{h}" for h in horizons], "tier"])


# ---------------------------------------------------------------- models


class ProbabilityModel:
    """A classifier whose `predict` returns the probability of the positive class (so it fits the artifact interface)."""

    def __init__(self, estimator: HistGradientBoostingClassifier):
        self.estimator = estimator

    def predict(self, X) -> np.ndarray:
        probabilities = self.estimator.predict_proba(X)
        classes = list(self.estimator.classes_)
        return probabilities[:, classes.index(1)] if 1 in classes else np.zeros(len(X))


def boosting_params(rows: int) -> dict:
    """Moderate settings: enough trees for the table size without taking an hour for the whole set."""
    return {
        "max_iter": 250,
        "learning_rate": 0.08,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 80 if rows > 50_000 else 20,
        "l2_regularization": 1.0,
        "early_stopping": True,
        "validation_fraction": 0.1,
        "n_iter_no_change": 15,
        "random_state": 0,
    }


def fit_quantile(X: pd.DataFrame, y: np.ndarray, quantile: float, categorical=None):
    model = HistGradientBoostingRegressor(
        loss="quantile", quantile=quantile, categorical_features=categorical, **boosting_params(len(X))
    )
    return model.fit(X, y)


def fit_classifier(X: pd.DataFrame, y: np.ndarray, categorical=None) -> ProbabilityModel:
    model = HistGradientBoostingClassifier(categorical_features=categorical, **boosting_params(len(X)))
    return ProbabilityModel(model.fit(X, y.astype(int)))


def subsample_players(frame: pd.DataFrame, max_rows: int, seed: int = 0) -> pd.DataFrame:
    """Keep whole players (all their rows) until about max_rows rows: seasons of one player stay together."""
    if len(frame) <= max_rows:
        return frame
    ids = frame["ID"].unique()
    rng = np.random.default_rng(seed)
    share = max_rows / len(frame)
    chosen = set(rng.choice(ids, size=max(1, int(len(ids) * share)), replace=False))
    return frame[frame["ID"].isin(chosen)]


def split_seasons(seasons: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """(train, test) seasons for a horizon: the test seasons are the latest ones with an observed future; the training seasons
    stop GAP seasons before them, so no training target is looked at by the test seasons."""
    observed = np.sort(np.unique(seasons))
    observed = observed[observed <= observed.max() - horizon]
    test = observed[-TEST_SEASONS:]
    train = observed[observed < test.min() - GAP]
    return train, test


# ---------------------------------------------------------------- metrics and baselines


def pinball(y: np.ndarray, prediction: np.ndarray, quantile: float) -> float:
    error = y - prediction
    return float(np.mean(np.maximum(quantile * error, (quantile - 1) * error)))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def log_loss(y: np.ndarray, p: np.ndarray, eps: float = 1e-6) -> float:
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def auc(y: np.ndarray, p: np.ndarray) -> float:
    positives = int(y.sum())
    negatives = len(y) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = pd.Series(p).rank().to_numpy()
    return float((ranks[y == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def calibration_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Mean prediction and observed rate in bins of equal numbers of rows (the last bin holds the most likely players)."""
    order = np.argsort(p)
    rows = []
    for chunk in np.array_split(order, bins):
        rows.append({"predicted": float(p[chunk].mean()), "observed": float(y[chunk].mean()), "n": len(chunk)})
    return pd.DataFrame(rows)


def age_slot_quantiles(train: pd.DataFrame, horizon: int, quantiles) -> dict:
    """Baseline: the empirical quantiles of the change by slot and age band in the training rows."""
    column = f"dvalue_h{horizon}"
    frame = train.dropna(subset=[column])
    frame = frame.assign(band=pd.cut(frame["Age"], [0, 19, 21, 23, 25, 27, 30, 34, 99], labels=False))
    table = frame.groupby(["slot", "band"])[column].quantile(list(quantiles)).unstack()
    overall = frame[column].quantile(list(quantiles))
    return {"table": table, "overall": overall}


def baseline_predict(baseline: dict, rows: pd.DataFrame, quantile: float) -> np.ndarray:
    band = pd.cut(rows["Age"], [0, 19, 21, 23, 25, 27, 30, 34, 99], labels=False)
    index = pd.MultiIndex.from_arrays([rows["slot"].to_numpy(), band.to_numpy()])
    table = baseline["table"].reindex(index)
    values = table[quantile].to_numpy() if quantile in table else np.full(len(rows), np.nan)
    return np.where(np.isnan(values), baseline["overall"][quantile], values)


# ---------------------------------------------------------------- artifacts


def artifact_name(kind: str, family: str, horizon: int, detail: str = "") -> str:
    suffix = f"-{detail}" if detail else ""
    return f"{OUTLOOK_DIR}/{kind}-{family}-h{horizon}{suffix}"


def make_artifact(name, model, features, target, save, ratings, seasons, metrics, notes) -> Artifact:
    return Artifact(
        name=name, model=model, features=list(features), target=target, save=save, ratings=ratings,
        family="hist-gradient-boosting", seasons=[int(s) for s in seasons], metrics=metrics, params={}, notes=notes,
    )  # fmt: skip


def store(artifact: Artifact) -> None:
    save_artifact(artifact)


def load(save: str, ratings: str, name: str) -> Artifact:
    return load_artifact(save, ratings, name)


# ---------------------------------------------------------------- training and evaluation


AGE_BANDS = {"16-19": (16, 19), "20-22": (20, 22), "23-26": (23, 26), "27-30": (27, 30), "31+": (31, 60)}
MAX_ROWS = 400_000
BAND_LEVEL = 0.8


@dataclass
class Run:
    """What every artifact of one training run records."""

    save: str
    ratings: str
    seasons: list
    max_rows: int = MAX_ROWS
    evaluate: bool = True
    log: object = print

    def store(self, name, model, features, target, metrics, notes) -> None:
        store(make_artifact(name, model, features, target, self.save, self.ratings, self.seasons, metrics, notes))


def categorical_mask(features: list[str]) -> list[bool]:
    return [name == "slot" for name in features]


def quantile_fits(train: pd.DataFrame, features, target: str, quantiles) -> dict:
    y = train[target].to_numpy(dtype=float)
    return {q: fit_quantile(train[features], y, q, categorical_mask(features)) for q in quantiles}


def predict_quantiles(models: dict, X: pd.DataFrame) -> dict:
    """Predictions of every quantile model, sorted per row so that no band is upside down."""
    quantiles = sorted(models)
    stacked = np.sort(np.column_stack([models[q].predict(X) for q in quantiles]), axis=1)
    return {q: stacked[:, i] for i, q in enumerate(quantiles)}


def band_shift(y: np.ndarray, lower: np.ndarray, upper: np.ndarray, level: float = BAND_LEVEL) -> float:
    """How far to widen a [lower, upper] band on both sides so that it covers `level` of the outcomes (conformalized quantile
    regression): the `level` quantile of how far the outcomes fall outside the band (negative when they sit well inside).
    """
    return float(np.quantile(np.maximum(lower - y, y - upper), level))


def coverage(y: np.ndarray, lower: np.ndarray, upper: np.ndarray, shift: float = 0.0) -> float:
    return float(((y >= lower - shift) & (y <= upper + shift)).mean())


def calibrated_band_check(test: pd.DataFrame, y: np.ndarray, predictions: dict) -> dict:
    """The widening is fitted on the first half of the test seasons and its coverage checked on the second half."""
    first = (test["season"] <= np.median(test["season"].to_numpy())).to_numpy()
    shift = band_shift(y[first], predictions[0.1][first], predictions[0.9][first])
    return {
        "shift": band_shift(y, predictions[0.1], predictions[0.9]),
        "shift_first_half": shift,
        "coverage_second_half_raw": coverage(y[~first], predictions[0.1][~first], predictions[0.9][~first]),
        "coverage_second_half_adjusted": coverage(y[~first], predictions[0.1][~first], predictions[0.9][~first], shift),
    }


def regression_metrics(test: pd.DataFrame, horizon: int, predictions: dict, baseline: dict) -> dict:
    """Median accuracy, pinball loss against the age-and-slot baseline, and the coverage of the bands (change of the value)."""
    y = test[f"dvalue_h{horizon}"].to_numpy(dtype=float)
    level = test["value_now"].to_numpy(dtype=float) + y
    median = predictions[0.5]
    age_median = baseline_predict(baseline, test, 0.5)
    sst = float(((level - level.mean()) ** 2).sum())
    metrics = {
        "n": int(len(test)),
        "r2": 1 - float(((y - median) ** 2).sum()) / sst,
        "r2_persistence": 1 - float((y**2).sum()) / sst,
        "r2_age_baseline": 1 - float(((y - age_median) ** 2).sum()) / sst,
        "mae": float(np.abs(y - median).mean()),
        "mae_persistence": float(np.abs(y).mean()),
        "mae_age_baseline": float(np.abs(y - age_median).mean()),
        "coverage_80": coverage(y, predictions[0.1], predictions[0.9]),
        "coverage_50": coverage(y, predictions[0.25], predictions[0.75]),
        "pinball": {},
        "pinball_age_baseline": {},
        "by_age": {},
        "band": calibrated_band_check(test, y, predictions),
    }
    for q, prediction in predictions.items():
        metrics["pinball"][q] = pinball(y, prediction, q)
        metrics["pinball_age_baseline"][q] = pinball(y, baseline_predict(baseline, test, q), q)
    ages = test["Age"].to_numpy()
    for label, (low, high) in AGE_BANDS.items():
        mask = (ages >= low) & (ages <= high)
        if mask.sum() >= 200:
            metrics["by_age"][label] = {
                "n": int(mask.sum()),
                "mae": float(np.abs(y[mask] - median[mask]).mean()),
                "mae_persistence": float(np.abs(y[mask]).mean()),
                "coverage_80": coverage(y[mask], predictions[0.1][mask], predictions[0.9][mask]),
            }
    return metrics


def fit_probability_calibration(y: np.ndarray, p: np.ndarray) -> dict:
    """Monotone map from predicted to observed chances (isotonic), kept as two arrays so it needs no sklearn to apply."""
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(p, y)
    return {"x": iso.X_thresholds_.tolist(), "y": iso.y_thresholds_.tolist()}


def apply_probability_calibration(p: np.ndarray, calibration: dict) -> np.ndarray:
    return np.interp(p, calibration["x"], calibration["y"])


def probability_calibration_check(test: pd.DataFrame, y: np.ndarray, p: np.ndarray) -> dict:
    """The calibration is fitted on the first half of the test seasons and scored on the second half."""
    first = (test["season"] <= np.median(test["season"].to_numpy())).to_numpy()
    calibrated = apply_probability_calibration(p[~first], fit_probability_calibration(y[first], p[first]))
    return {
        "base_rate": float(y[~first].mean()),
        "mean_raw": float(p[~first].mean()),
        "mean_calibrated": float(calibrated.mean()),
        "brier_raw": brier(y[~first], p[~first]),
        "brier_calibrated": brier(y[~first], calibrated),
    }


def classifier_metrics(test: pd.DataFrame, target: str, probability: np.ndarray, train: pd.DataFrame) -> dict:
    """Ranking and calibration of a tier classifier against the rate by slot and by decile of today's percentile."""
    y = test[target].to_numpy(dtype=float)
    bins = np.linspace(0, 1, 11)

    def key(frame):
        decile = np.clip(np.digitize(frame["pct_now"].to_numpy(), bins) - 1, 0, 9)
        return pd.MultiIndex.from_arrays([frame["slot"].to_numpy(), decile])

    train_decile = np.clip(np.digitize(train["pct_now"].to_numpy(), bins) - 1, 0, 9)
    rates = train.groupby([train["slot"], train_decile])[target].mean()
    base = rates.reindex(key(test)).fillna(float(train[target].mean())).to_numpy()
    return {
        "n": int(len(test)),
        "base_rate": float(y.mean()),
        "auc": auc(y, probability),
        "auc_baseline": auc(y, base),
        "brier": brier(y, probability),
        "brier_baseline": brier(y, base),
        "log_loss": log_loss(y, probability),
        "log_loss_baseline": log_loss(y, base),
        "mean_prediction": float(probability.mean()),
        "calibration": calibration_table(y, probability).round(5).to_dict("records"),
    }


def train_value_models(run: Run, family: str, table: pd.DataFrame, horizon: int, train_seasons, test_seasons) -> dict:
    features = feature_names(family)
    column = f"dvalue_h{horizon}"
    known = table[table[column].notna()]
    metrics: dict = {}
    if run.evaluate:
        train = subsample_players(known[known["season"].isin(train_seasons)], run.max_rows)
        test = known[known["season"].isin(test_seasons)]
        models = quantile_fits(train, features, column, VALUE_QUANTILES)
        predictions = predict_quantiles(models, test[features])
        metrics = regression_metrics(test, horizon, predictions, age_slot_quantiles(train, horizon, VALUE_QUANTILES))
    final = subsample_players(known[known["season"] <= max(run.seasons) - horizon], run.max_rows)
    for q, model in quantile_fits(final, features, column, VALUE_QUANTILES).items():
        notes = {"kind": "value", "family": family, "horizon": horizon, "quantile": q}
        if metrics:
            notes["band_shift"] = metrics["band"]["shift"]  # widen [q10, q90] by this much for an 80 % band
        run.store(artifact_name("value", family, horizon, f"q{int(q * 100)}"), model, features, column, metrics, notes)
    return metrics


def train_tier_models(run: Run, family: str, table: pd.DataFrame, horizon: int, train_seasons, test_seasons) -> dict:
    features = feature_names(family)
    results = {}
    for tier in ("top50", "top10"):
        column = f"{tier}_h{horizon}"
        labelled = table[table[column].notna()]
        metrics: dict = {}
        calibration = None
        if run.evaluate:
            train = subsample_players(labelled[labelled["season"].isin(train_seasons)], run.max_rows)
            test = labelled[labelled["season"].isin(test_seasons)]
            model = fit_classifier(train[features], train[column].to_numpy(), categorical_mask(features))
            probability = model.predict(test[features])
            metrics = classifier_metrics(test, column, probability, train)
            y = test[column].to_numpy(dtype=float)
            metrics["calibration_check"] = probability_calibration_check(test, y, probability)
            calibration = fit_probability_calibration(y, probability)
        final = subsample_players(labelled, run.max_rows)
        model = fit_classifier(final[features], final[column].to_numpy(), categorical_mask(features))
        notes = {"kind": tier, "family": family, "horizon": horizon, "calibration": calibration}
        run.store(artifact_name(tier, family, horizon), model, features, column, metrics, notes)
        results[tier] = metrics
    return results


def train_family_horizon(run: Run, family: str, table: pd.DataFrame, horizon: int) -> dict:
    """Value quantiles and top-10 / top-50 classifiers of one family and horizon: hold-out metrics, then final models."""
    train_seasons, test_seasons = split_seasons(table["season"].to_numpy(), horizon)
    metrics = {
        "family": family,
        "horizon": horizon,
        "train_seasons": [int(train_seasons.min()), int(train_seasons.max())],
        "test_seasons": [int(test_seasons.min()), int(test_seasons.max())],
    }
    metrics["value"] = train_value_models(run, family, table, horizon, train_seasons, test_seasons)
    metrics.update(train_tier_models(run, family, table, horizon, train_seasons, test_seasons))
    if run.evaluate:
        value, top50 = metrics["value"], metrics["top50"]
        run.log(
            f"  {family} h{horizon}: value R2 {value['r2']:.3f} (persistence {value['r2_persistence']:.3f}), "
            f"80% band {value['coverage_80']:.2f} -> {value['band']['coverage_second_half_adjusted']:.2f} calibrated, "
            f"top50 AUC {top50['auc']:.3f}"
        )
    return metrics


def train_stay_model(run: Run, table: pd.DataFrame, horizon: int, train_seasons, test_seasons) -> dict:
    column = f"stay_h{horizon}"
    labelled = table[table[column].notna()]
    metrics: dict = {}
    train = subsample_players(labelled[labelled["season"].isin(train_seasons)], run.max_rows)
    test = labelled[labelled["season"].isin(test_seasons)]
    if run.evaluate and len(train) and len(test):
        model = fit_classifier(train[PLAYER_FEATURES], train[column].to_numpy())
        probability = model.predict(test[PLAYER_FEATURES])
        y = test[column].to_numpy(dtype=float)
        by_age = train.groupby(np.clip(train["Age"], 15, 40))[column].mean()
        base = by_age.reindex(np.clip(test["Age"], 15, 40)).fillna(float(train[column].mean())).to_numpy()
        metrics = {
            "n": int(len(test)),
            "base_rate": float(y.mean()),
            "auc": auc(y, probability),
            "brier": brier(y, probability),
            "brier_baseline": brier(y, base),
            "calibration": calibration_table(y, probability).round(5).to_dict("records"),
        }
    final = subsample_players(labelled, run.max_rows)
    model = fit_classifier(final[PLAYER_FEATURES], final[column].to_numpy())
    run.store(
        artifact_name("stay", "player", horizon),
        model,
        PLAYER_FEATURES,
        column,
        metrics,
        {"kind": "stay", "horizon": horizon},
    )
    return metrics


def train_availability_models(run: Run, table: pd.DataFrame, horizon: int, train_seasons, test_seasons) -> dict:
    target = f"avail_h{horizon}"
    known = table[table[target].notna()]
    metrics: dict = {}
    if known.empty:  # a save without playing-time data (an export with one season, or fixtures)
        run.log(f"  no playing time known for availability h{horizon}: skipped")
        return metrics
    train = subsample_players(known[known["season"].isin(train_seasons)], run.max_rows)
    test = known[known["season"].isin(test_seasons)]
    if run.evaluate and len(train) and len(test):
        predictions = predict_quantiles(
            quantile_fits(train, AVAIL_FEATURES, target, AVAIL_QUANTILES), test[AVAIL_FEATURES]
        )
        y = test[target].to_numpy(dtype=float)
        constant = float(train[target].median())
        by_tier = train.groupby("Prone3_ord")[target].median().reindex(test["Prone3_ord"]).fillna(constant).to_numpy()
        predicted_by_tier = test.assign(p=predictions[0.5]).groupby("Prone3_ord")["p"].mean()
        metrics = {
            "n": int(len(test)),
            "mae": float(np.abs(y - predictions[0.5]).mean()),
            "mae_constant": float(np.abs(y - constant).mean()),
            "mae_tier_median": float(np.abs(y - by_tier).mean()),
            "coverage_80": coverage(y, predictions[0.1], predictions[0.9]),
            "median_by_prone": {int(k): float(v) for k, v in predicted_by_tier.items()},
        }
    final = subsample_players(known, run.max_rows)
    for q, model in quantile_fits(final, AVAIL_FEATURES, target, AVAIL_QUANTILES).items():
        notes = {"kind": "avail", "horizon": horizon, "quantile": q}
        run.store(
            artifact_name("avail", "player", horizon, f"q{int(q * 100)}"), model, AVAIL_FEATURES, target, metrics, notes
        )
    return metrics


def train_player_models(run: Run, table: pd.DataFrame, horizon: int) -> dict:
    """Stay classifier and availability quantiles (player level) for one horizon."""
    train_seasons, test_seasons = split_seasons(table["season"].to_numpy(), horizon)
    metrics = {
        "horizon": horizon,
        "stay": train_stay_model(run, table, horizon, train_seasons, test_seasons),
        "avail": train_availability_models(run, table, horizon, train_seasons, test_seasons),
    }
    if run.evaluate and metrics["avail"] and metrics["stay"]:
        stay, avail = metrics["stay"], metrics["avail"]
        run.log(
            f"  player h{horizon}: stay AUC {stay['auc']:.3f}, availability MAE {avail['mae']:.3f} "
            f"(constant {avail['mae_constant']:.3f}, tier median {avail['mae_tier_median']:.3f})"
        )
    return metrics


def train_outlook_models(
    save: str,
    ratings: str,
    families=("hitters", "pitchers"),
    horizons=HORIZONS,
    max_rows: int = MAX_ROWS,
    evaluate: bool = True,
    log=print,
) -> dict:
    """Train and store every outlook model of a save; returns the hold-out metrics by family and horizon."""
    tables = outlook_data.read_tables(save)
    seasons = sorted(tables["players"]["season"].unique().tolist())
    run = Run(save, ratings, [int(s) for s in seasons], max_rows, evaluate, log)
    results: dict = {"families": {}, "player": {}, "seasons": run.seasons}
    for family in families:
        table = family_table(save, family, horizons, tables)
        log(f"{family}: {len(table)} rows")
        for h in horizons:
            results["families"][(family, h)] = train_family_horizon(run, family, table, h)
    ptable = add_player_targets(player_table(tables["players"], tables["values"]), horizons)
    for h in horizons:
        results["player"][h] = train_player_models(run, ptable, h)
    return results

"""Park effects on a hitter, from his ratings: how often he hits home runs, doubles, triples and hits at all in a neutral park.

The value models predict park-adjusted (neutral park) outcomes. To value a player in a given park the viewer needs how his
production would change there: home runs, doubles, triples and hits each move with their own park factor (the left or right one
by the side he bats from), and singles are what is left of the hits. So four small models predict, from the ratings, his
neutral-park rate per plate appearance of each of those events. They are trained like the pipeline's park adjustment works: each
count divided by the effective park factor the player faced (the ParkHR, ParkD, ParkT and ParkAVG columns of the hitting
output). The artifacts are stored next to the value models in a `park` folder. See viewer/parks.py for how the rates are turned
into runs.
"""

import numpy as np
import pandas as pd

from model import artifact, data, domains
from model.converters import SWITCH
from model.artifact import Artifact
from model.metrics import summarize
from model.train import fit_table, make_model

EVENTS = {"HR": ("HR", "ParkHR"), "DOUBLE": ("DOUBLE", "ParkD"), "TRIPLE": ("TRIPLE", "ParkT"), "H": ("H", "ParkAVG")}
FEATURES = list(dict.fromkeys([*domains.REVIEWED_FEATURES[domains.BATTING], "SPE", "STE", "RUN"]))
FOLDER = "park"
MIN_PA = domains.MIN_PA


def event_table(save: str, seasons, features=FEATURES, min_pa: int = MIN_PA) -> pd.DataFrame:
    """Per player and season: the neutral-park rate per PA of each event (columns of EVENTS), the PA as `weight`, the ratings."""
    frames = []
    for season in [int(s) for s in seasons]:
        hitting = data.read_output(save, season, "hitting")
        hitting = hitting[hitting["PA"] >= min_pa]
        frame = pd.DataFrame({"ID": hitting["ID"], "season": season, "weight": hitting["PA"].astype(float)})
        for event, (count, factor) in EVENTS.items():
            frame[event] = hitting[count] / hitting[factor] / hitting["PA"]
        frames.append(frame.join(data.read_ratings(save, season, features), on="ID"))
    table = pd.concat(frames, ignore_index=True).replace([np.inf, -np.inf], np.nan).dropna()
    return table[table["weight"] > 0].reset_index(drop=True)


def train(save: str, ratings: str, seasons, seed: int = 0, holdout: int = 4) -> dict[str, Artifact]:
    """One ridge model per event on the seasons; the last `holdout` seasons are scored (weighted R^2) when there are enough."""
    seasons = [int(s) for s in seasons]
    table = event_table(save, seasons)
    held = min(holdout, len(set(table["season"])) - 1)
    made = {}
    for event in EVENTS:
        view = data.TrainingTable(f"{save}-park-{event}", table.rename(columns={event: "target"}), list(FEATURES))
        metrics = {"rows": len(view), "mean_rate": float(np.average(view.y, weights=view.w))}
        if held >= 1:
            cutoff = sorted(set(table["season"]))[-held]
            train_rows = np.flatnonzero(view.frame["season"] < cutoff)
            test_rows = np.flatnonzero(view.frame["season"] >= cutoff)
            fitted = fit_table(make_model("ridge"), view, train_rows, seed)
            test = view.frame.iloc[test_rows]
            metrics["holdout"] = summarize(test["target"], fitted.predict(test[view.features]), test["weight"])
        model = fit_table(make_model("ridge"), view, seed=seed)
        made[event] = Artifact(
            name=f"{FOLDER}/{event}", model=model, features=list(FEATURES), target=f"{event} per PA, neutral park", save=save,
            ratings=ratings, family="ridge", seasons=seasons, metrics=metrics, notes={"stream": "park", "event": event},
        )  # fmt: skip
    return made


def load_models(models_from: str, ratings: str) -> dict[str, Artifact]:
    """The park models of a model set (FileNotFoundError when it has none: train_models.py --streams park)."""
    folder = artifact.models_dir(models_from, ratings) / FOLDER
    if not folder.exists():
        raise FileNotFoundError(f"{folder} not found; run scripts/train_models.py {models_from} --streams park")
    return {event: artifact.load_artifact(models_from, ratings, f"{FOLDER}/{event}") for event in EVENTS}


def rates(ratings: pd.DataFrame, models: dict[str, Artifact]) -> pd.DataFrame:
    """Neutral-park rate per PA of each event for every player of `ratings` (numeric, indexed by ID) with all the ratings; NaN for
    the others. Rates are kept between 0 and 1."""
    out = pd.DataFrame(index=ratings.index, columns=list(models), dtype=float)
    complete = ratings[FEATURES].notna().all(axis=1)
    for event, loaded in models.items():
        if complete.any():
            out.loc[complete, event] = np.clip(loaded.predict(ratings[complete]), 0.0, 1.0)
    return out


def left_share(bats: pd.Series, switch_left: float) -> pd.Series:
    """The share of a batter's plate appearances from the left side: lefties 1, righties 0 (the codes of the batting hand column
    of player-data: 1 right, 2 left, 3 switch), switch hitters `switch_left` (the league's share of PA against right-handed
    pitchers).
    """
    share = pd.Series(0.5, index=bats.index)
    share[bats == 1] = 0.0
    share[bats == 2] = 1.0
    share[bats == SWITCH] = switch_left
    return share

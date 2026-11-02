"""The shipped models: which artifacts exist, how each is trained, and how a set of ratings is scored.

Training and scoring share the feature code in model/data.py. Primary models are the stacked platoon models for batting
and for the two pitching parts (value against each side, blended with the platoon mix of the hand), plus baserunning and
one fielding model per position. The plain total models for batting and pitching stay as fallbacks (plans/14-models.md).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from model import data, domains, evaluate
from model.artifact import Artifact
from model.metrics import summarize
from model.train import fit_table, holdout_last_seasons, make_model

STACKED_STREAMS = (domains.BATTING, domains.PITCHING_FIP, domains.PITCHING_BABIP)
SPLIT_SHARED = {domains.BATTING: [], domains.PITCHING_FIP: ["STM", "PIT"], domains.PITCHING_BABIP: []}


@dataclass(frozen=True)
class Spec:
    name: str
    stream: str
    kind: str  # "plain" or "split"
    family: str
    position: int | None = None


def family_of(stream: str, position: int | None = None) -> str:
    chosen = domains.CHOSEN_FAMILY[stream]
    return chosen[position] if isinstance(chosen, dict) else chosen


def specs(streams=None, include_plain: bool = True) -> list[Spec]:
    """Every artifact to train, split models first for each stream."""
    wanted = list(streams) if streams else [*domains.REVIEWED_FEATURES]
    unknown = [s for s in wanted if s not in domains.REVIEWED_FEATURES]
    if unknown:
        raise ValueError(f"unknown streams {unknown}; known: {list(domains.REVIEWED_FEATURES)}")
    out = []
    for stream in wanted:
        if stream == domains.FIELDING:
            out += [Spec(f"fielding-{p}", stream, "plain", family_of(stream, p), p) for p in domains.FIELD_POSITIONS]
            continue
        if stream in STACKED_STREAMS:
            out.append(Spec(f"{stream}-split", stream, "split", family_of(stream)))
        if stream not in STACKED_STREAMS or include_plain:
            out.append(Spec(stream, stream, "plain", family_of(stream)))
    return out


def build_table(spec: Spec, save: str, seasons):
    reviewed = domains.REVIEWED_FEATURES
    if spec.stream == domains.BATTING:
        if spec.kind == "split":
            return data.batting_split_table(save, seasons, shared=SPLIT_SHARED[spec.stream])
        return data.batting_table(save, seasons, reviewed[spec.stream])
    if spec.stream == domains.BASERUNNING:
        return data.baserunning_table(save, seasons, reviewed[spec.stream])
    if spec.stream == domains.PITCHING_FIP:
        if spec.kind == "split":
            return data.pitching_split_table(save, seasons, "fip", shared=SPLIT_SHARED[spec.stream])
        return data.pitching_fip_table(save, seasons, reviewed[spec.stream])
    if spec.stream == domains.PITCHING_BABIP:
        if spec.kind == "split":
            return data.pitching_split_table(save, seasons, "babip", shared=SPLIT_SHARED[spec.stream])
        return data.pitching_babip_table(save, seasons, reviewed[spec.stream])
    return data.fielding_table(save, seasons, spec.position, reviewed[domains.FIELDING][spec.position])


def hand_values(stream: str, features: pd.DataFrame) -> np.ndarray:
    """Batting: 0 right, 1 left, 2 switch. Pitching: 0 throws right, 1 throws left."""
    if stream == domains.BATTING:
        return np.where(features["bats_switch"] == 1, 2.0, features["bats_left"]).astype(float)
    return features["throws_left"].to_numpy(dtype=float)


def babip_conversion(save: str, seasons) -> dict:
    """League balls in play per 200 IP and the run value of a hit over an out, to turn a BABIP difference into runs."""
    bip, innings, values = 0.0, 0.0, []
    for season in seasons:
        pitching = data.read_output(save, season, "pitching")
        bip += pitching["BIP"].sum()
        innings += pitching["IPClean"].sum()
        calcs = data.read_output(save, season, "woba-calcs").set_index("Stat")["Value"]
        totals = data.read_output(save, season, "batting-totals").iloc[0]
        hits = totals["1B"] + totals["2B"] + totals["3B"]
        values.append(
            sum(totals[b] * (calcs[f"run_value_{b}"] + calcs["RUNS_MINUS"]) for b in ["1B", "2B", "3B"]) / hits
        )
    return {"bip_per_200_ip": bip / innings * 200, "hit_value": float(np.mean(values))}


def model_params(spec: Spec, device: str | None) -> dict:
    return {"device": device} if spec.family == "mlp" and device else {}


def train_spec(
    spec: Spec,
    save: str,
    ratings: str,
    seasons,
    cv: bool = True,
    device: str | None = None,
    seed: int = 0,
    folds: int = 5,
    holdout: int = 5,
) -> Artifact:
    """Fit one model on all the rows of the seasons; with cv also store honest player-grouped scores."""
    seasons = [int(s) for s in seasons]
    table = build_table(spec, save, seasons)
    params = model_params(spec, device)
    metrics = {"rows": len(table), "players": int(table.frame[table.groups].nunique())}
    if cv:
        fold_of_player = evaluate.player_folds(table.frame[table.groups], folds, seed)
        prediction = evaluate.cross_validate_folds(spec.family, table, fold_of_player, folds, seed, **params)
        metrics["cv"] = summarize(table.y, prediction, table.w)
        held_out = min(holdout, len(seasons) - 1)  # train on at least one season; none to hold out with a single one
        if held_out >= 1:
            _, metrics["holdout"] = holdout_last_seasons(spec.family, table, held_out, seed, **params)
            metrics["holdout_seasons"] = held_out
    model = fit_table(make_model(spec.family, **params), table, seed=seed)
    notes = {"stream": spec.stream, "kind": spec.kind, "position": spec.position}
    if spec.kind == "split":
        frame = table.frame
        hand = hand_values(spec.stream, frame)
        notes["share_right"] = evaluate.platoon_mix(frame.assign(hand=hand), "hand")
        notes["shared"] = SPLIT_SHARED[spec.stream]
    if spec.stream == domains.PITCHING_BABIP:
        notes["conversion"] = babip_conversion(save, seasons)
    return Artifact(
        name=spec.name,
        model=model,
        features=list(table.features),
        target=table.target,
        save=save,
        ratings=ratings,
        family=spec.family,
        seasons=seasons,
        metrics=metrics,
        params=params,
        notes=notes,
    )


# ---------------------------------------------------------------- scoring


def required_rating_columns(artifacts: dict[str, Artifact]) -> list[str]:
    """Rating columns of player-data that the artifacts read."""
    columns: list[str] = []
    for loaded in artifacts.values():
        notes = loaded.notes
        if notes["kind"] == "plain":
            columns += loaded.features
        elif notes["stream"] == domains.BATTING:
            columns += data.batting_split_rating_columns(notes["shared"])
        else:
            columns += data.pitching_split_rating_columns(notes["shared"])
    return list(dict.fromkeys(columns))


def _predict(loaded: Artifact, features: pd.DataFrame) -> pd.Series:
    """Prediction for the rows with every feature present; the others get NaN."""
    ok = features[loaded.features].notna().all(axis=1)
    out = pd.Series(np.nan, index=features.index)
    if ok.any():
        out[ok] = loaded.predict(features[ok])
    return out


def score(artifacts: dict[str, Artifact], ratings: pd.DataFrame) -> pd.DataFrame:
    """Predicted values for every player in `ratings` (numeric, indexed by ID), one column per quantity.

    Runs per 600 PA (batting, baserunning), per 200 IP (pitching) and per season (fielding, one column per position).
    Stacked models give `<stream>_vs_right`, `<stream>_vs_left` and the total blended with the platoon mix of the hand;
    the plain models of the stacked streams are added as `<stream>_plain` fallbacks."""
    out = pd.DataFrame(index=ratings.index)
    for name, loaded in artifacts.items():
        notes = loaded.notes
        stream = notes["stream"]
        if notes["kind"] == "plain":
            column = name.replace("-", "_")
            out[column + ("_plain" if stream in STACKED_STREAMS else "")] = _predict(loaded, ratings)
            continue
        make = data.batting_side_features if stream == domains.BATTING else data.pitching_side_features
        sides = {side: make(ratings, side, notes["shared"]) for side in domains.SIDES}
        right = _predict(loaded, sides[domains.RIGHT])
        left = _predict(loaded, sides[domains.LEFT])
        hand = pd.Series(hand_values(stream, sides[domains.RIGHT]), index=ratings.index)
        share = hand.map(notes["share_right"]).astype(float)
        base = stream.replace("-", "_")
        out[f"{base}_vs_right"] = right
        out[f"{base}_vs_left"] = left
        out[base] = share * right + (1 - share) * left
    babip = artifacts.get(f"{domains.PITCHING_BABIP}-split")
    if babip is not None:
        conversion = babip.notes["conversion"]
        out["pitching_babip_runs"] = evaluate.babip_to_runs(
            out["pitching_babip"], conversion["bip_per_200_ip"], conversion["hit_value"]
        )
        if "pitching_fip" in out:
            out["pitching_runs"] = out["pitching_fip"] + out["pitching_babip_runs"]
    return out

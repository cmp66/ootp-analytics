"""Stage 4 of plans/14-models.md: train the value streams on the reviewed attributes and compare.

For every stream:
  - ridge, gradient boosting and the network on the reviewed attribute list and on the old list, scored with
    player-grouped 5-fold cross-validation (shared folds) and a hold-out of the last seasons (weighted R^2, MAE);
    the simplest family within 0.003 of the best is chosen;
  - batting, pitching FIP and pitching BABIP: the stacked platoon model (rows per side, blended with the league's
    platoon mix) against the plain total model, scored on the total target for the same player-seasons, and how well it
    predicts the gap between a player's two sides;
  - pitching as a whole: FIP part + BABIP part (in runs per 200 IP) against the runs those parts actually produced and
    against WAA200;
  - calibration by decile for the chosen models.

Usage: PYTHONPATH=. python scripts/compare_models.py [save] [--start 2046 --stop 2086] [--streams ...] [--no-old]
       [--out plans/data/model-comparison.txt]
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from model import data, domains, evaluate
from model.metrics import calibration, summarize
from model.train import holdout_last_seasons

FAMILIES = ["ridge", "boosting", "mlp"]
PITCH_SHARED = ["STM", "PIT"]
BATTING_SHARED = []


def plain_tables(save, seasons, wanted, lists):
    """(name, table) for each wanted stream with the attribute lists in `lists` (reviewed or baseline)."""
    if domains.BATTING in wanted:
        yield "batting", data.batting_table(save, seasons, lists[domains.BATTING])
    if domains.BASERUNNING in wanted:
        yield "baserunning", data.baserunning_table(save, seasons, lists[domains.BASERUNNING])
    if domains.PITCHING_FIP in wanted:
        yield "pitching-fip", data.pitching_fip_table(save, seasons, lists[domains.PITCHING_FIP])
    if domains.PITCHING_BABIP in wanted:
        yield "pitching-babip", data.pitching_babip_table(save, seasons, ["PBABIP vR", "PBABIP vL"])
    if domains.FIELDING in wanted:
        for position in domains.FIELD_POSITIONS:
            yield f"fielding-{position}", data.fielding_table(
                save, seasons, position, lists[domains.FIELDING][position]
            )


def score_plain(name, table, label, rows, chosen_scores):
    folds = evaluate.player_folds(table.frame[table.groups], 5, 0)
    scores = {}
    for family in FAMILIES:
        prediction = evaluate.cross_validate_folds(family, table, folds)
        cv = summarize(table.y, prediction, table.w)
        _, held = holdout_last_seasons(family, table, 5)
        rows.append(
            {
                "stream": name,
                "list": label,
                "family": family,
                "rows": len(table),
                "cv_r2": cv["r2"],
                "cv_mae": cv["mae"],
                "holdout_r2": held["r2"],
                "holdout_mae": held["mae"],
            }
        )
        scores[family] = cv["r2"]
    chosen_scores[(name, label)] = scores


def side_gap_correlation(split_table, blended):
    """Weighted correlation of the predicted right-minus-left gap with the actual gap of the two side targets."""
    frame = split_table.frame
    right = frame[frame["side_left"] == 0].set_index(["ID", "season"])
    left = frame[frame["side_left"] == 1].set_index(["ID", "season"])
    both = right[["target", "weight"]].join(left[["target", "weight"]], lsuffix="_r", rsuffix="_l", how="inner")
    merged = blended.set_index(["ID", "season"]).join(both, how="inner")
    gap_actual = merged["target_r"] - merged["target_l"]
    gap_pred = merged["right"] - merged["left"]
    weight = 1 / (1 / merged["weight_r"] + 1 / merged["weight_l"])  # the less certain side limits the gap
    mean_a, mean_p = np.average(gap_actual, weights=weight), np.average(gap_pred, weights=weight)
    cov = np.average((gap_actual - mean_a) * (gap_pred - mean_p), weights=weight)
    spread = np.sqrt(
        np.average((gap_actual - mean_a) ** 2, weights=weight) * np.average((gap_pred - mean_p) ** 2, weights=weight)
    )
    return cov / spread if spread > 0 else float("nan")


def stacked_comparison(save, seasons, kind, rows):
    """Plain total model versus stacked platoon model for batting, pitching FIP or pitching BABIP."""
    reviewed = domains.REVIEWED_FEATURES
    if kind == "batting":
        total = data.batting_table(save, seasons, reviewed[domains.BATTING])
        split = data.batting_split_table(save, seasons, shared=BATTING_SHARED)
        hand = "bats_left"
        split.frame["hand"] = np.where(split.frame["bats_switch"] == 1, 2, split.frame["bats_left"])
        hand = "hand"
    else:
        part = "fip" if kind == "pitching-fip" else "babip"
        if part == "fip":
            total = data.pitching_fip_table(save, seasons, reviewed[domains.PITCHING_FIP])
            split = data.pitching_split_table(save, seasons, "fip", shared=PITCH_SHARED)
        else:
            total = data.pitching_babip_table(save, seasons, reviewed[domains.PITCHING_BABIP])
            split = data.pitching_split_table(save, seasons, "babip")
        hand = "throws_left"
    mix = evaluate.platoon_mix(split.frame, hand)
    folds = evaluate.player_folds(pd.concat([total.frame["ID"], split.frame["ID"]]), 5, 0)
    for family in FAMILIES:
        plain = evaluate.cross_validate_folds(family, total, folds)
        merged, stacked_scores = evaluate.cross_validate_stacked(family, split, total, folds, mix, hand)
        plain_frame = total.frame[["ID", "season", total.target, total.weight]].assign(plain=plain)
        both = merged.merge(plain_frame[["ID", "season", "plain"]], on=["ID", "season"])
        plain_scores = summarize(both[total.target], both["plain"], both[total.weight])
        stacked_same = summarize(both[total.target], both["total"], both[total.weight])
        rows.append(
            {
                "stream": kind,
                "family": family,
                "player_seasons": len(both),
                "plain_r2": plain_scores["r2"],
                "stacked_r2": stacked_same["r2"],
                "plain_mae": plain_scores["mae"],
                "stacked_mae": stacked_same["mae"],
                "side_gap_corr": side_gap_correlation(split, merged),
            }
        )
    return mix


def pitching_combined(save, seasons, fip_family, babip_family, text):
    """FIP part + BABIP part predictions (runs per 200 IP) against the runs those parts produced and WAA200."""
    reviewed = domains.REVIEWED_FEATURES
    fip = data.pitching_fip_table(save, seasons, reviewed[domains.PITCHING_FIP])
    babip = data.pitching_babip_table(save, seasons, reviewed[domains.PITCHING_BABIP])
    folds = evaluate.player_folds(pd.concat([fip.frame["ID"], babip.frame["ID"]]), 5, 0)
    fip_pred = evaluate.cross_validate_folds(fip_family, fip, folds)
    babip_pred = evaluate.cross_validate_folds(babip_family, babip, folds)
    truth = []
    for season in seasons:
        p = data.read_output(save, season, "pitching")
        p["season"] = season
        truth.append(p[["ID", "season", "IPClean", "BIP", "FIPRunsAdj", "BABIPRuns", "WAA"]])
    truth = pd.concat(truth, ignore_index=True)
    calcs = data.read_output(save, seasons[len(seasons) // 2], "woba-calcs").set_index("Stat")["Value"]
    totals = data.read_output(save, seasons[len(seasons) // 2], "batting-totals").iloc[0]
    hits = totals["1B"] + totals["2B"] + totals["3B"]
    hit_value = sum(totals[b] * (calcs[f"run_value_{b}"] + calcs["RUNS_MINUS"]) for b in ["1B", "2B", "3B"]) / hits
    bip_per_200 = (truth["BIP"].sum() / truth["IPClean"].sum()) * 200
    frame = (
        fip.frame[["ID", "season"]]
        .assign(fip=fip_pred)
        .merge(
            babip.frame[["ID", "season"]].assign(babip=evaluate.babip_to_runs(babip_pred, bip_per_200, hit_value)),
            on=["ID", "season"],
        )
    )
    frame = frame.merge(truth, on=["ID", "season"])
    frame = frame[frame["IPClean"] >= domains.MIN_IP]
    frame["pred"] = frame["fip"] + frame["babip"]
    weight = frame["IPClean"]
    actual_parts = (frame["FIPRunsAdj"] + frame["BABIPRuns"]) / frame["IPClean"] * 200
    waa200 = frame["WAA"] * 10 / frame["IPClean"] * 200
    lines = [f"pitching combined ({fip_family} FIP part + {babip_family} BABIP part), {len(frame)} player-seasons"]
    for label, y, pred in [
        ("FIP part vs FIPRunsAdj/200", frame["FIPRunsAdj"] / frame["IPClean"] * 200, frame["fip"]),
        ("FIP + BABIP vs FIPRuns+BABIPRuns /200", actual_parts, frame["pred"]),
        ("FIP only vs FIPRuns+BABIPRuns /200", actual_parts, frame["fip"]),
        ("FIP + BABIP vs WAA200", waa200, frame["pred"]),
        ("FIP only vs WAA200", waa200, frame["fip"]),
    ]:
        s = summarize(y, pred, weight)
        lines.append(f"  {label:42s} R2 {s['r2']:.4f}  MAE {s['mae']:.3f}")
    text.append("\n".join(lines))


def calibration_text(name, table, family, mix_text):
    folds = evaluate.player_folds(table.frame[table.groups], 5, 0)
    prediction = evaluate.cross_validate_folds(family, table, folds)
    rows = calibration(table.y, prediction, table.w, 10)
    body = " ".join(f"{p:+.1f}/{a:+.1f}" for p, a, _ in rows)
    mix_text.append(f"{name} ({family}) decile mean prediction/actual: {body}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("save", nargs="?", default="Model-Sim")
    parser.add_argument("--start", type=int, default=2046)
    parser.add_argument("--stop", type=int, default=2086)
    parser.add_argument("--streams", nargs="+", default=[*domains.BASELINE_FEATURES, "pitching-babip"])
    parser.add_argument("--no-old", action="store_true", help="skip the old attribute lists")
    parser.add_argument("--out", default="plans/data/model-comparison.txt")
    args = parser.parse_args()
    logging.getLogger("model.data").setLevel(logging.ERROR)
    seasons = list(range(args.start, args.stop + 1))
    rows, chosen_scores, text = [], {}, []

    lists = [("reviewed", domains.REVIEWED_FEATURES)] + ([] if args.no_old else [("old", domains.BASELINE_FEATURES)])
    tables = {}
    for label, feature_lists in lists:
        for name, table in plain_tables(args.save, seasons, args.streams, feature_lists):
            if label == "old" and name == "pitching-babip":
                continue
            score_plain(name, table, label, rows, chosen_scores)
            if label == "reviewed":
                tables[name] = table
            print(f"{label} {name} done", flush=True)
    plain = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    text.append(
        "== plain models, player-grouped 5-fold CV and last-5-seasons holdout (weighted)\n"
        + plain.round(4).to_string(index=False)
    )

    chosen = {name: evaluate.choose_family(chosen_scores[(name, "reviewed")]) for name in tables}
    text.append("\n== chosen family (simplest within 0.003 CV R^2 of the best): " + str(chosen))

    stacked_rows = []
    for kind in [s for s in ("batting", "pitching-fip", "pitching-babip") if s in args.streams]:
        mix = stacked_comparison(args.save, seasons, kind, stacked_rows)
        text.append(f"platoon mix for {kind}: share against right {mix}")
        print(f"stacked {kind} done", flush=True)
    if stacked_rows:
        text.append(
            "\n== stacked platoon model vs plain total model (same player-seasons, total target)\n"
            + pd.DataFrame(stacked_rows).round(4).to_string(index=False)
        )

    if "pitching-fip" in tables and "pitching-babip" in tables:
        pitching_combined(args.save, seasons, chosen["pitching-fip"], chosen["pitching-babip"], text)

    cal = []
    for name, table in tables.items():
        if name in ("batting", "baserunning", "pitching-fip"):
            calibration_text(name, table, chosen[name], cal)
    text.append("\n== calibration\n" + "\n".join(cal))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(text))
    print("\n".join(text))


if __name__ == "__main__":
    main()

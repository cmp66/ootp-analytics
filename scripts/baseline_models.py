"""Stage 2 of plans/14-models.md: how good are ratings-to-value models under an honest evaluation?

For each value stream, with the attribute lists of the old model classes, compares ridge regression, gradient boosting
and the neural network (new trainer) with
  - player-grouped k-fold cross-validation (no player on both sides),
  - a hold-out of the last seasons (train on the early seasons, predict later ones),
and reports the old protocol's number for the network (random 50 % of the rows, weights chosen on that same half) to
show how optimistic it was. Weighted R^2 and MAE: rows count by the volume behind their target.

Usage: PYTHONPATH=. python scripts/baseline_models.py [save] [--start 2046 --stop 2086] [--families ridge boosting mlp]
       [--domains batting baserunning pitching-fip pitching-babip fielding] [--folds 5] [--holdout 5]
"""

import argparse
import logging
import time

import numpy as np
import pandas as pd

from model import data, domains
from model.metrics import summarize
from model.train import MLPConfig, MLPModel, cross_validate, holdout_last_seasons

logging.getLogger("model.data").setLevel(logging.ERROR)


def tables_for(save: str, seasons: range, wanted: list[str]):
    base = domains.BASELINE_FEATURES
    if domains.BATTING in wanted:
        yield "batting", data.batting_table(save, seasons, base[domains.BATTING])
    if domains.BASERUNNING in wanted:
        yield "baserunning", data.baserunning_table(save, seasons, base[domains.BASERUNNING])
    if domains.PITCHING_FIP in wanted:
        yield "pitching-fip", data.pitching_fip_table(save, seasons, base[domains.PITCHING_FIP])
    if domains.PITCHING_BABIP in wanted:
        yield "pitching-babip", data.pitching_babip_table(save, seasons, ["PBABIP vR", "PBABIP vL"])
    if domains.FIELDING in wanted:
        for position in domains.FIELD_POSITIONS:
            yield f"fielding-{position}", data.fielding_table(save, seasons, position, base[domains.FIELDING][position])


def old_protocol(table, seed: int = 0) -> dict:
    """Random half of the rows for stopping and for the reported error, like the old RegressionRunner."""
    model = MLPModel(MLPConfig(val_fraction=0.5, seed=seed))
    frame = table.frame
    model.fit(frame[table.features], frame[table.target], frame[table.weight], np.arange(len(frame)), seed=seed)
    rows = frame.iloc[model.val_index]
    return summarize(rows[table.target], model.predict(rows[table.features]), rows[table.weight])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("save", nargs="?", default="Model-Sim")
    parser.add_argument("--start", type=int, default=2046)
    parser.add_argument("--stop", type=int, default=2086)
    parser.add_argument("--families", nargs="+", default=["ridge", "boosting", "mlp"])
    parser.add_argument("--domains", nargs="+", default=[d for d in domains.BASELINE_FEATURES] + ["pitching-babip"])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--holdout", type=int, default=5, help="number of last seasons held out")
    parser.add_argument("--no-old-protocol", action="store_true")
    args = parser.parse_args()

    rows = []
    for name, table in tables_for(args.save, range(args.start, args.stop + 1), args.domains):
        started = time.perf_counter()
        for family in args.families:
            _, cv = cross_validate(family, table, args.folds)
            _, held = holdout_last_seasons(family, table, args.holdout)
            row = {
                "stream": name,
                "rows": len(table),
                "family": family,
                "cv_r2": cv["r2"],
                "cv_mae": cv["mae"],
                "holdout_r2": held["r2"],
                "holdout_mae": held["mae"],
            }
            if family == "mlp" and not args.no_old_protocol:
                row["old_protocol_r2"] = old_protocol(table)["r2"]
            rows.append(row)
        print(f"{name}: {len(table)} rows, {time.perf_counter() - started:.0f}s", flush=True)
    result = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print()
    print(result.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

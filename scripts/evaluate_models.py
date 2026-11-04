"""Score trained models against the real outcomes of some seasons.

Usage:
    python scripts/evaluate_models.py Model-Sim                           # every model on the seasons it was trained on
    python scripts/evaluate_models.py Model-Sim --start 2081 --stop 2086  # seasons the models never saw (train with --stop 2080)
    python scripts/evaluate_models.py Model-Sim --streams batting --calibration
    python scripts/evaluate_models.py Dugout26 --models-from Model-Sim-26-Std   # another save's outcomes, same ratings set

Prints, for each artifact, the scores stored at training time (player-grouped cross-validation and hold-out, honest)
and the scores on the seasons asked for (in-sample when those seasons were trained on, marked "in-sample"). Stacked
platoon models are also scored on the total value, the blend of the two sides against the total target. With
--calibration the mean prediction and mean outcome are printed for ten bins of the prediction.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


from model import artifact, domains, evaluate, registry, scales  # noqa: E402
from model.metrics import calibration, summarize  # noqa: E402
from util.config import get_save  # noqa: E402
from util.seasons import available_seasons, select_seasons  # noqa: E402


def spec_of(loaded) -> registry.Spec:
    notes = loaded.notes
    return registry.Spec(loaded.name, notes["stream"], notes["kind"], loaded.family, notes["position"])


def evaluate_one(loaded, save: str, seasons: list[int], show_calibration: bool) -> list[str]:
    spec = spec_of(loaded)
    try:
        table = registry.build_table(spec, save, seasons)
    except KeyError, FileNotFoundError:
        if spec.kind != "split":
            raise
        return [f"{loaded.name}: skipped, {save} has no vs-right / vs-left stats for these seasons"]
    prediction = loaded.predict(table.frame[loaded.features])
    scores = summarize(table.y, prediction, table.w)
    in_sample = loaded.save == save and bool(set(seasons) & set(loaded.seasons))
    stored = loaded.metrics
    lines = [
        f"{loaded.name} ({loaded.family}) trained on {loaded.seasons[0]}-{loaded.seasons[-1]}, {stored.get('rows', '?')} rows"
    ]
    if "cv" in stored:
        held = stored.get("holdout")
        lines.append(
            f"    stored  cv R2 {stored['cv']['r2']:.4f} MAE {stored['cv']['mae']:.3f}"
            + (f" | holdout R2 {held['r2']:.4f} MAE {held['mae']:.3f}" if held else "")
        )
    lines.append(
        f"    {seasons[0]}-{seasons[-1]} ({'in-sample' if in_sample else 'unseen'}) {len(table)} rows: "
        f"R2 {scores['r2']:.4f} MAE {scores['mae']:.3f} RMSE {scores['rmse']:.3f}"
    )
    if spec.kind == "split":
        total = registry.build_table(registry.Spec(spec.stream, spec.stream, "plain", loaded.family), save, seasons)
        frame = table.frame.assign(hand=registry.hand_values(spec.stream, table.frame))
        blended = evaluate.blend_sides(frame, prediction, loaded.notes["share_right"], frame["hand"])
        merged = blended.merge(
            total.frame[["ID", "season", total.target, total.weight]], on=["ID", "season"], how="inner"
        )
        total_scores = summarize(merged[total.target], merged["total"], merged[total.weight])
        lines.append(
            f"    total value (blend of the two sides) {len(merged)} player-seasons: "
            f"R2 {total_scores['r2']:.4f} MAE {total_scores['mae']:.3f}"
        )
    if show_calibration:
        rows = calibration(table.y, prediction, table.w, 10)
        lines.append("    calibration (prediction/actual): " + " ".join(f"{p:+.2f}/{a:+.2f}" for p, a, _ in rows))
    return lines


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("save")
    parser.add_argument("--start", type=int, help="first season to score (default: the first with data)")
    parser.add_argument("--stop", type=int, help="last season to score (default: the latest)")
    parser.add_argument("--streams", nargs="+", choices=list(domains.REVIEWED_FEATURES), help="default: all")
    parser.add_argument("--calibration", action="store_true")
    parser.add_argument("--models-from", help="save whose models to score (default: the same save; same ratings set)")
    args = parser.parse_args(argv)
    try:
        save = get_save(args.save)
        available = available_seasons(args.save)
        seasons = select_seasons(
            available, args.start if args.start is not None else available[0], args.stop or available[-1]
        )
        models_from = args.models_from or args.save
        if get_save(models_from).ratings != save.ratings:
            raise ValueError(
                f"{models_from} models are for ratings {get_save(models_from).ratings!r}, {args.save} uses {save.ratings!r}"
            )
        folder = artifact.models_dir(models_from, scales.model_set(save.ratings, save.scales))
        names = sorted(p.stem for p in folder.glob("*.joblib")) if folder.exists() else []
        if not names:
            raise FileNotFoundError(f"no trained models in {folder}; run scripts/train_models.py {models_from}")
        artifacts = [
            artifact.load_artifact(models_from, scales.model_set(save.ratings, save.scales), name) for name in names
        ]
    except (KeyError, ValueError, FileNotFoundError, IndexError) as error:
        print(f"error: {error.args[0] if isinstance(error, KeyError) else error}", file=sys.stderr)
        return 2
    for loaded in artifacts:
        if args.streams and loaded.notes["stream"] not in args.streams:
            continue
        print("\n".join(evaluate_one(loaded, args.save, seasons, args.calibration)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Train the value models of a save and store one artifact per model under files/models/<save>/<ratings>/.

Usage:
    python scripts/train_models.py Model-Sim --start 2046 --stop 2086            # every stream
    python scripts/train_models.py Model-Sim --start 2046 --streams batting baserunning
    python scripts/train_models.py Model-Sim --start 2046 --no-cv --device cuda  # skip the honest scores; GPU for networks

Streams: batting, baserunning, pitching-fip, pitching-babip, fielding. Batting and the two pitching parts get a stacked
platoon model (`<stream>-split`, the primary one) and a plain total model (the fallback). Each artifact holds the fitted
model with its scaler, the feature names, the seasons it was trained on and, unless --no-cv, player-grouped
cross-validation and last-seasons hold-out scores. The ratings set comes from config/saves.toml. Training uses the CPU
for the linear and boosting models and for the small networks unless --device says otherwise (OOTP_DEVICE also works).
"""

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model import artifact, data, domains, park, registry, scales  # noqa: E402
from util.config import get_save  # noqa: E402
from util.seasons import available_seasons, select_seasons  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("save", help="save name (folder under files/ and table in config/saves.toml)")
    parser.add_argument("--start", type=int, help="first season to train on (default: the first with data)")
    parser.add_argument("--stop", type=int, help="last season to train on (default: the latest)")
    parser.add_argument(
        "--streams", nargs="+", choices=[*domains.REVIEWED_FEATURES, "park"],
        help="default: all; park = the event rates of the park adjustment (model/park.py)",
    )  # fmt: skip
    parser.add_argument("--no-plain", action="store_true", help="skip the plain total fallbacks of the stacked streams")
    parser.add_argument("--no-cv", action="store_true", help="skip the cross-validation and hold-out scores")
    parser.add_argument("--device", help="torch device for networks (cpu, cuda, xpu); default: OOTP_DEVICE or CPU")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--scales",
        metavar="LEAGUE",
        help="train on ratings turned to the mix of scales of this league (its 1-10 columns); the models go to "
        "<ratings>@<league> and the league names them in the `scales` key of config/saves.toml",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    try:
        save = get_save(args.save)
        available = available_seasons(args.save)
        if not available:
            raise ValueError(f"no seasons with data for {args.save}; run scripts/process_seasons.py first")
        start = args.start if args.start is not None else available[0]
        stop = args.stop if args.stop is not None else available[-1]
        seasons = select_seasons(available, start, stop)
        value_streams = [s for s in args.streams if s != "park"] if args.streams else None
        train_park = args.streams is None or "park" in args.streams
        todo = (
            [] if args.streams and not value_streams else registry.specs(value_streams, include_plain=not args.no_plain)
        )
        view, key, coarse = None, save.ratings, []
        if args.scales:
            conversion = scales.Conversion.load(
                scales.version_of(save.ratings), relative=get_save(args.scales).relative_ratings
            )
            coarse = scales.profile_columns(args.scales, conversion=conversion)
            if not coarse:
                raise ValueError(f"{args.scales} has no ratings on the 1-10 scale: nothing to convert")
            view, key = scales.coarse_view(conversion, coarse, seed=args.seed), scales.model_set(
                save.ratings, args.scales
            )
    except (KeyError, ValueError) as error:
        print(f"error: {error.args[0] if isinstance(error, KeyError) else error}", file=sys.stderr)
        return 2

    print(f"Training {len(todo)} models on {args.save} {seasons[0]}-{seasons[-1]} (ratings {key})")
    if args.scales:
        print(f"  ratings turned to 1-10 as in {args.scales}: {len(coarse)} columns")
    failed = []
    for spec in todo:
        started = time.perf_counter()
        try:
            with data.ratings_view(view):
                trained = registry.train_spec(
                    spec, args.save, key, seasons, cv=not args.no_cv, device=args.device, seed=args.seed
                )
            if args.scales:
                trained.notes["scales"] = {"profile": args.scales, "columns": coarse}
            path = artifact.save_artifact(trained)
        except FileNotFoundError as error:
            print(f"  {spec.name}: {error}", file=sys.stderr)
            failed.append(spec.name)
            continue
        metrics = trained.metrics
        scores = f", cv R2 {metrics['cv']['r2']:.3f}" if "cv" in metrics else ""
        if "holdout" in metrics:
            scores += f", holdout R2 {metrics['holdout']['r2']:.3f}"
        print(
            f"  {spec.name} ({spec.family}): {metrics['rows']} rows{scores} -> {path.name} "
            f"[{time.perf_counter() - started:.0f}s]",
            flush=True,
        )
    if train_park:
        started = time.perf_counter()
        try:
            with data.ratings_view(view):
                made = park.train(args.save, key, seasons, seed=args.seed)
            for event, trained in made.items():
                if args.scales:
                    trained.notes["scales"] = {"profile": args.scales, "columns": coarse}
                artifact.save_artifact(trained)
            scores = ", ".join(
                f"{e} R2 {m.metrics['holdout']['r2']:.2f}" for e, m in made.items() if "holdout" in m.metrics
            )
            print(
                f"  park event rates: {len(made)} models, hold-out {scores} [{time.perf_counter() - started:.0f}s]",
                flush=True,
            )
        except FileNotFoundError as error:
            print(f"  park: {error}", file=sys.stderr)
            failed.append("park")
    if failed:
        print("Not trained (missing output files): " + ", ".join(failed), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

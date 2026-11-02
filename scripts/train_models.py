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

from model import artifact, domains, registry  # noqa: E402
from util.config import get_save  # noqa: E402
from util.seasons import available_seasons, select_seasons  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("save", help="save name (folder under files/ and table in config/saves.toml)")
    parser.add_argument("--start", type=int, help="first season to train on (default: the first with data)")
    parser.add_argument("--stop", type=int, help="last season to train on (default: the latest)")
    parser.add_argument("--streams", nargs="+", choices=list(domains.REVIEWED_FEATURES), help="default: all")
    parser.add_argument("--no-plain", action="store_true", help="skip the plain total fallbacks of the stacked streams")
    parser.add_argument("--no-cv", action="store_true", help="skip the cross-validation and hold-out scores")
    parser.add_argument("--device", help="torch device for networks (cpu, cuda, xpu); default: OOTP_DEVICE or CPU")
    parser.add_argument("--seed", type=int, default=0)
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
        todo = registry.specs(args.streams, include_plain=not args.no_plain)
    except (KeyError, ValueError) as error:
        print(f"error: {error.args[0] if isinstance(error, KeyError) else error}", file=sys.stderr)
        return 2

    print(f"Training {len(todo)} models on {args.save} {seasons[0]}-{seasons[-1]} (ratings {save.ratings})")
    failed = []
    for spec in todo:
        started = time.perf_counter()
        try:
            trained = registry.train_spec(
                spec, args.save, save.ratings, seasons, cv=not args.no_cv, device=args.device, seed=args.seed
            )
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
    if failed:
        print("Not trained (missing output files): " + ", ".join(failed), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

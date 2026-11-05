"""One-button update after new ratings or statistics files (the human leagues of config/update.toml).

Usage:
    PYTHONPATH=. python scripts/update_all.py                  # every save of config/update.toml, latest season
    PYTHONPATH=. python scripts/update_all.py TGS27              # one save
    PYTHONPATH=. python scripts/update_all.py --if-changed     # skip saves whose input files are unchanged
    PYTHONPATH=. python scripts/update_all.py --steps          # show the steps and exit
    PYTHONPATH=. python scripts/update_all.py TGS27 --only outlook lists --season 2046

For each save the steps run in order (a failed step stops that save, the other saves still run):
    process   processed output files of the season (scripts/process_seasons.py)
    values    predicted values of every player, plan 14 models (scripts/predict_values.py)
    park      neutral-park event rates of the hitters, for the park adjustment of the viewer (scripts/predict_park.py)
    outlook   floor / ceiling, top-10 / top-50 chances, injury cost (scripts/predict_outlook.py)
    lists     every list of config/viewer.toml as a csv in output/lists/ (scripts/outlook_lists.py)
When a function or output is added to the project, add its step to STEPS here (and to docs/updating-data.md) so the update
stays complete.
Training is not part of the update: trained models only change when you retrain them (train_models.py, train_outlook_models.py).
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model import park, scales  # noqa: E402
from scripts import outlook_lists, predict_outlook, predict_park, predict_values, process_seasons  # noqa: E402
from util import paths  # noqa: E402
from util.config import get_save  # noqa: E402
from util.update_config import UPDATE_CONFIG, load_update_config  # noqa: E402,F401
from util.seasons import available_seasons  # noqa: E402
from viewer import queries  # noqa: E402

STAMP = "update-stamp.json"


@dataclass(frozen=True)
class Job:
    save: str
    season: int
    models_from: str
    org: str | None


@dataclass(frozen=True)
class Step:
    name: str
    description: str
    run: Callable[[Job], int]


def _process(job: Job) -> int:
    return process_seasons.main([job.save, "--season", str(job.season)])


def _values(job: Job) -> int:
    return predict_values.main([job.save, "--season", str(job.season), "--models-from", job.models_from])


def _park(job: Job) -> int:
    """The neutral-park event rates for the park adjustment of the viewer; skipped when the model set has no park models yet."""
    config = get_save(job.save)
    try:
        park.load_models(job.models_from, scales.model_set(config.ratings, config.scales))
    except FileNotFoundError:
        print("  skipped: no park models (train them with scripts/train_models.py <training save> --streams park)")
        return 0
    return predict_park.main([job.save, "--season", str(job.season), "--models-from", job.models_from])


def _outlook(job: Job) -> int:
    return predict_outlook.main([job.save, "--season", str(job.season), "--models-from", job.models_from])


def _lists(job: Job) -> int:
    """Every list of config/viewer.toml to output/lists/; the lists that need an organization are skipped without one."""
    defaults, specs = queries.load_list_specs()
    out_dir = paths.output_dir(job.save, job.season) / "lists"
    out_dir.mkdir(parents=True, exist_ok=True)
    status = 0
    for name, spec in specs.items():
        needs_org = spec.metric in ("p_start", "p_roster") or spec.pool in ("org", "others")
        if needs_org and job.org is None:
            print(f"  list {name}: skipped (set org in config/update.toml)")
            continue
        target = out_dir / f"{job.save}-{job.season}-{name}.csv"
        argv = [job.save, "--list", name, "--season", str(job.season), "--out", str(target)]
        if job.org is not None:
            argv += ["--org", job.org]
        status |= outlook_lists.main(argv)
    return status


STEPS = [
    Step("process", "processed output files of the season", _process),
    Step("values", "predicted values (plan 14 models)", _values),
    Step("park", "neutral-park event rates for the park adjustment of the viewer", _park),
    Step("outlook", "outlook scores: floor, ceiling, tier chances, injury cost", _outlook),
    Step("lists", "the lists of config/viewer.toml as csv files", _lists),
]


def input_signature(save: str, season: int) -> dict[str, list]:
    """Size and modification time of every raw file of the season (ratings and statistics) and of the ballpark file."""
    config = get_save(save)
    root = paths.export_dir(save, season) if config.source == "export" else paths.dump_dir(save, config.ratings, season)
    files = [p for p in sorted(root.rglob("*.csv")) if "output" not in p.relative_to(root).parts]
    return {str(p.relative_to(root)): [p.stat().st_size, int(p.stat().st_mtime)] for p in files}


def stamp_file(save: str, season: int) -> Path:
    return paths.output_dir(save, season) / STAMP


def unchanged(save: str, season: int) -> bool:
    try:
        return json.loads(stamp_file(save, season).read_text()) == input_signature(save, season)
    except OSError, ValueError:
        return False


def run_save(job: Job, steps: list[Step]) -> list[tuple[str, str, float]]:
    """Run the steps of one save in order; a failing step stops the save. Returns (step, outcome, seconds)."""
    results = []
    for step in steps:
        print(f"\n== {job.save} {job.season}: {step.name} ({step.description})")
        started = time.time()
        try:
            code = step.run(job)
            outcome = "ok" if code == 0 else f"failed (exit {code})"
        except Exception as error:  # a broken step must not stop the other saves
            outcome = f"failed ({type(error).__name__}: {error})"
        results.append((step.name, outcome, time.time() - started))
        if outcome != "ok":
            break
    return results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("saves", nargs="*", help="saves to update (default: all of config/update.toml)")
    parser.add_argument("--season", type=int, help="default: the latest season with data")
    parser.add_argument("--only", nargs="+", metavar="STEP", help="run only these steps")
    parser.add_argument("--if-changed", action="store_true", help="skip a save whose input files have not changed")
    parser.add_argument("--steps", action="store_true", help="list the steps and exit")
    args = parser.parse_args(argv)
    if args.steps:
        for step in STEPS:
            print(f"{step.name:10s} {step.description}")
        return 0
    known = {s.name for s in STEPS}
    if args.only and set(args.only) - known:
        print(f"error: unknown step(s) {sorted(set(args.only) - known)}; known: {sorted(known)}", file=sys.stderr)
        return 2
    config = load_update_config()
    saves = args.saves or list(config)
    if set(saves) - set(config):
        print(
            f"error: {sorted(set(saves) - set(config))} not in config/update.toml (known: {list(config)})",
            file=sys.stderr,
        )
        return 2
    failed = False
    summary = []
    for save in saves:
        entry = config[save]
        try:
            get_save(save)
            season = args.season if args.season is not None else (available_seasons(save) or [None])[-1]
            if season is None:
                raise ValueError(f"no seasons with data for {save}")
        except (KeyError, ValueError) as error:
            print(f"error: {save}: {error}", file=sys.stderr)
            failed = True
            continue
        if args.if_changed and unchanged(save, season):
            print(f"{save} {season}: input files unchanged, skipped")
            continue
        wanted = args.only or entry.get("steps") or [s.name for s in STEPS]
        steps = [s for s in STEPS if s.name in wanted]
        job = Job(save, season, entry.get("models_from", save), entry.get("org"))
        results = run_save(job, steps)
        summary += [(save, season, *r) for r in results]
        if all(r[1] == "ok" for r in results) and not args.only:
            stamp_file(save, season).write_text(json.dumps(input_signature(save, season)))
        else:
            failed |= any(r[1] != "ok" for r in results)
    print("\nSummary")
    for save, season, step, outcome, seconds in summary:
        print(f"  {save} {season} {step:8s} {outcome} ({seconds:.0f}s)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

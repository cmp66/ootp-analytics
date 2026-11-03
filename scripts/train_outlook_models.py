"""Train the outlook models of a save (plan 16, stage 3) and write the evaluation report.

Usage:
    PYTHONPATH=. python scripts/train_outlook_models.py Model-Sim                     # every model, evaluation included
    PYTHONPATH=. python scripts/train_outlook_models.py Model-Sim --families hitters --horizons 1 3
    PYTHONPATH=. python scripts/train_outlook_models.py Model-Sim --no-eval           # final models only (faster)

Needs the tables of scripts/build_outlook_tables.py. Artifacts go to files/models/<save>/<ratings>/outlook/ and the hold-out
evaluation (latest seasons, five-season gap, compared with persistence and age baselines) to
docs/outlook-model-evaluation-<save>.md. See model/outlook_models.py for what is trained.
"""

import argparse
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model import outlook_models  # noqa: E402
from util.config import get_save  # noqa: E402


def md_row(*cells) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def pct(x) -> str:
    return f"{x:.0%}"


def skill(model: float, baseline: float) -> str:
    """Gain over a baseline loss (positive = better); n/a when the baseline is perfect."""
    return "n/a" if not baseline else f"{1 - model / baseline:+.0%}"


def value_section(results: dict) -> list[str]:
    lines = [
        "## Value: where the rating-based value of a slot is going",
        "",
        "Quantile models of the change of the slot value h seasons ahead, for players ranked at the slot in both seasons. R² is",
        "of the value h seasons ahead. Baselines: persistence (value stays) and the median change of the same slot and age band.",
        "Pinball skill is the gain in pinball loss over the age-band baseline for the 10th, 50th and 90th percentile.",
        "The 80 % band is [10th, 90th]; calibrated = widened by the shift fitted on the first half of the test seasons and",
        "checked on the second half.",
        "",
        "| Family | h | Test rows | R² | Persistence | Age baseline | MAE | MAE persistence | MAE age baseline | "
        "Skill q10 / q50 / q90 | 80 % band | calibrated | 50 % band |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    for (family, h), metrics in sorted(results["families"].items()):
        v = metrics["value"]
        skills = " / ".join(skill(v["pinball"][q], v["pinball_age_baseline"][q]) for q in (0.1, 0.5, 0.9))
        lines.append(
            md_row(
                family, h, v["n"], f"{v['r2']:.3f}", f"{v['r2_persistence']:.3f}", f"{v['r2_age_baseline']:.3f}",
                f"{v['mae']:.2f}", f"{v['mae_persistence']:.2f}", f"{v['mae_age_baseline']:.2f}", skills,
                pct(v["coverage_80"]), pct(v["band"]["coverage_second_half_adjusted"]), pct(v["coverage_50"]),
            )
        )  # fmt: skip
    lines += ["", "By age band (horizon 3): median error against persistence, and the raw 80 % band coverage:", ""]
    lines += ["| Family | Age | Rows | MAE | MAE persistence | 80 % band |", "|---|---|---:|---:|---:|---:|"]
    for (family, h), metrics in sorted(results["families"].items()):
        if h != 3:
            continue
        for age, a in metrics["value"]["by_age"].items():
            lines.append(
                md_row(family, age, a["n"], f"{a['mae']:.2f}", f"{a['mae_persistence']:.2f}", pct(a["coverage_80"]))
            )
    return lines + [""]


def tier_section(results: dict) -> list[str]:
    lines = [
        "## Tier chances: top 10 and top 50 at the slot",
        "",
        "Classifiers for being ranked top 10 / top 50 at the same slot h seasons ahead (a player who is gone counts as no). The",
        "baseline is the observed rate by slot and decile of today's percentile. Brier skill is the gain over that baseline.",
        "",
        "| Family | h | Tier | Test rows | Base rate | Mean prediction | AUC | AUC baseline | Brier skill | Log-loss skill |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for (family, h), metrics in sorted(results["families"].items()):
        for tier in ("top50", "top10"):
            t = metrics[tier]
            lines.append(
                md_row(
                    family, h, tier, t["n"], f"{t['base_rate']:.3%}", f"{t['mean_prediction']:.3%}", f"{t['auc']:.3f}",
                    f"{t['auc_baseline']:.3f}", skill(t["brier"], t["brier_baseline"]),
                    skill(t["log_loss"], t["log_loss_baseline"]),
                )
            )  # fmt: skip
    lines += [
        "",
        "The chances are a little low in absolute terms in the latest seasons (the share of future top players rises over time),",
        "so the stored classifiers carry an isotonic calibration fitted on the test rows. Check: fitted on the first half of the",
        "test seasons, scored on the second half:",
        "",
        "| Family | h | Tier | Observed rate | Mean raw | Mean calibrated | Brier raw | Brier calibrated |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for (family, h), metrics in sorted(results["families"].items()):
        for tier in ("top50", "top10"):
            c = metrics[tier]["calibration_check"]
            lines.append(
                md_row(
                    family, h, tier, f"{c['base_rate']:.3%}", f"{c['mean_raw']:.3%}", f"{c['mean_calibrated']:.3%}",
                    f"{c['brier_raw']:.5f}", f"{c['brier_calibrated']:.5f}",
                )
            )  # fmt: skip
    lines += ["", "Calibration of the top-50 chance at horizon 3 (rows sorted by prediction, ten equal groups):", ""]
    for (family, h), metrics in sorted(results["families"].items()):
        if h != 3:
            continue
        lines += [f"{family}:", "", "| Group | Predicted | Observed |", "|---:|---:|---:|"]
        for i, row in enumerate(metrics["top50"]["calibration"], 1):
            lines.append(md_row(i, f"{row['predicted']:.4f}", f"{row['observed']:.4f}"))
        lines.append("")
    return lines


def player_section(results: dict) -> list[str]:
    lines = [
        "## Staying in the league and availability",
        "",
        "| h | Stay: rows | Base rate | AUC | Brier skill vs age | Availability: rows | MAE | MAE constant | "
        "MAE tier median | 80 % band |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for h, metrics in sorted(results["player"].items()):
        s, a = metrics["stay"], metrics["avail"]
        if not a or not s:
            continue
        lines.append(
            md_row(
                h, s["n"], pct(s["base_rate"]), f"{s['auc']:.3f}", skill(s["brier"], s["brier_baseline"]), a["n"],
                f"{a['mae']:.3f}", f"{a['mae_constant']:.3f}", f"{a['mae_tier_median']:.3f}", pct(a["coverage_80"]),
            )
        )  # fmt: skip
    lines += [
        "",
        "Median availability predicted for players of each proneness tier (0 durable, 1 normal, 2 fragile; test seasons):",
        "",
    ]
    lines += ["| h | Durable | Normal | Fragile |", "|---:|---:|---:|---:|"]
    for h, metrics in sorted(results["player"].items()):
        if not metrics["avail"]:
            continue
        m = metrics["avail"]["median_by_prone"]
        lines.append(md_row(h, *[f"{m.get(i, float('nan')):.3f}" for i in (0, 1, 2)]))
    return lines + [""]


def build_report(save: str, ratings: str, results: dict, seconds: float) -> str:
    seasons = results["seasons"]
    one = next(iter(results["families"].values()))
    lines = [
        f"# Outlook model evaluation: {save} ({ratings})",
        "",
        f"Generated {date.today()} by `scripts/train_outlook_models.py` from seasons {seasons[0]}-{seasons[-1]} "
        "of the long CPU league "
        f"(trained in {seconds / 60:.0f} minutes). Each game version has its own models.",
        "",
        "Hold-out: the test rows are the latest six seasons with an observed future; training uses rows at least ten seasons "
        "earlier "
        f"(for horizon 1 the training seasons end {one['train_seasons'][1]}, the test seasons are {one['test_seasons'][0]}-"
        f"{one['test_seasons'][1]}; the dates move with the horizon). The final models are retrained on all observed rows.",
        "",
        *value_section(results),
        *tier_section(results),
        *player_section(results),
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("save")
    parser.add_argument(
        "--families", nargs="+", choices=list(outlook_models.FAMILIES), default=list(outlook_models.FAMILIES)
    )
    parser.add_argument(
        "--horizons", nargs="+", type=int, default=list(outlook_models.OUTLOOK_HORIZONS),
        help="seasons ahead; 99 is the career view (best season within the next 10)",
    )  # fmt: skip
    parser.add_argument(
        "--max-rows", type=int, default=outlook_models.MAX_ROWS, help="training rows per model (whole players)"
    )
    parser.add_argument("--no-eval", action="store_true", help="train the final models only")
    parser.add_argument("--out", help="report (default docs/outlook-model-evaluation-<save>.md)")
    args = parser.parse_args(argv)
    try:
        ratings = get_save(args.save).ratings
        started = time.time()
        results = outlook_models.train_outlook_models(
            args.save, ratings, tuple(args.families), tuple(args.horizons), args.max_rows, not args.no_eval
        )
    except (KeyError, ValueError, FileNotFoundError) as error:
        print(f"error: {error.args[0] if isinstance(error, KeyError) else error}", file=sys.stderr)
        return 2
    if not args.no_eval:
        out = Path(args.out) if args.out else Path("docs") / f"outlook-model-evaluation-{args.save}.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(build_report(args.save, ratings, results, time.time() - started))
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

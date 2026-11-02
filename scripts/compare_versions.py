"""Compare the attribute impacts of two model sets (for example OOTP 26 and OOTP 27) and write a ranked report.

For every shipped model the permutation importance and the effect of +10 points / +1 SD are computed for both saves with
the method of scripts/rating_importance.py. The report ranks the attributes by how much their effect on seasonal value
changed, per model and overall, and shows how the value of a player with the same ratings differs between the two model
sets (one set of ratings is scored with both).

Usage: PYTHONPATH=. python scripts/compare_versions.py [--old Model-Sim-26-Std] [--new Model-Sim] [--ratings-from Model-Sim
       --season 2060] [--folds 5] [--repeats 3] [--out docs/ootp26-vs-ootp27-attribute-impact.md]
"""

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from model import artifact, domains, registry, relative  # noqa: E402
from scripts import predict_values, rating_importance  # noqa: E402
from scripts.rating_importance import LABELS, POSITIONS, UNITS  # noqa: E402
from util.config import get_save  # noqa: E402

ORDER = ["batting-split", "baserunning", "pitching-fip-split", "pitching-babip-split"] + [
    f"fielding-{p}" for p in domains.FIELD_POSITIONS
]
TITLES = {
    "batting-split": "Batting",
    "baserunning": "Baserunning",
    "pitching-fip-split": "Pitching: FIP part",
    "pitching-babip-split": "Pitching: BABIP part",
}


def title_of(loaded) -> str:
    return TITLES.get(loaded.name, f"Fielding at {POSITIONS.get(loaded.notes['position'], '?')}")


def label(feature: str) -> str:
    return f"{LABELS.get(feature, feature)} (`{feature}`)"


def impact(save: str, name: str, folds: int, repeats: int, seed: int = 0):
    """Importance table of one shipped model of `save`, effects in the units of its stream (BABIP part in runs)."""
    ratings = get_save(save).ratings
    loaded = artifact.load_artifact(save, ratings, name)
    table = rating_importance.importance_for(loaded, save, folds, repeats, seed)
    divisor = 1.0
    if loaded.notes["stream"] == domains.PITCHING_BABIP:
        conversion = loaded.notes["conversion"]
        divisor = -1.0 / (conversion["bip_per_200_ip"] * conversion["hit_value"])
    table = table.assign(effect_step=table["effect_step"] / divisor, effect_sd=table["effect_sd"] / divisor)
    return loaded, table


def merge_impacts(old: pd.DataFrame, new: pd.DataFrame, noise: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per rating of either model; `_old` / `_new` columns and the changes (new minus old)."""
    both = old.set_index("feature").add_suffix("_old").join(new.set_index("feature").add_suffix("_new"), how="outer")
    both["step_change"] = both["effect_step_new"].fillna(0) - both["effect_step_old"].fillna(0)
    both["sd_change"] = both["effect_sd_new"].fillna(0) - both["effect_sd_old"].fillna(0)
    both["share_change"] = both["share_new"].fillna(0) - both["share_old"].fillna(0)
    if noise is not None:
        both["sd_noise"] = noise.set_index("feature")["sd_noise"].reindex(both.index)
    return both.reset_index()


def seed_noise(first: pd.DataFrame, second: pd.DataFrame) -> pd.DataFrame:
    """How much the +1 SD effect moves when only the folds and scrambles change (a floor for what is a real change)."""
    joined = (
        first.set_index("feature")["effect_sd"].to_frame("a").join(second.set_index("feature")["effect_sd"].rename("b"))
    )
    return pd.DataFrame({"feature": joined.index, "sd_noise": (joined["a"] - joined["b"]).abs().to_numpy()})


def fmt(value, digits=2, sign=True):
    if pd.isna(value):
        return "-"
    return f"{value:+.{digits}f}" if sign else f"{value:.{digits}f}"


def pct(value):
    return "-" if pd.isna(value) else f"{value:.0%}"


def ratio(new, old):
    if pd.isna(new) or pd.isna(old) or abs(old) < 0.05:
        return "-"
    return f"{new / old - 1:+.0%}"


def same_ratings_comparison(old_save, new_save, ratings_from, season) -> pd.DataFrame:
    """Score one set of ratings with both model sets: weighted mean value, difference and rank correlation per column."""
    old_ratings, new_ratings = get_save(old_save).ratings, get_save(new_save).ratings
    old_models = predict_values.load_artifacts(old_save, old_ratings)
    new_models = predict_values.load_artifacts(new_save, new_ratings)
    players = registry_players(ratings_from, season, {**old_models, **new_models})
    scored = {
        "old": registry.score(old_models, players),
        "new": registry.score(new_models, players),
    }
    weights = relative.season_weights(ratings_from, season)
    rows = []
    for column in scored["new"].columns:
        family = relative.family_of(column)
        if family is None or column.endswith(("_vs_right", "_vs_left", "_plain")) or column not in scored["old"]:
            continue
        pair = pd.concat([scored["old"][column].rename("old"), scored["new"][column].rename("new")], axis=1)
        pair = pair.join(weights[family].rename("w"), how="inner").dropna()
        pair = pair[pair["w"] > 0]
        if len(pair) < 20:
            continue
        w = pair["w"].to_numpy()
        mean_old, mean_new = np.average(pair["old"], weights=w), np.average(pair["new"], weights=w)
        rows.append(
            {
                "column": column,
                "players": len(pair),
                "mean_old": mean_old,
                "mean_new": mean_new,
                "difference": mean_new - mean_old,
                "sd_old": float(np.sqrt(np.average((pair["old"] - mean_old) ** 2, weights=w))),
                "sd_new": float(np.sqrt(np.average((pair["new"] - mean_new) ** 2, weights=w))),
                "rank_corr": float(pair["old"].rank().corr(pair["new"].rank())),
            }
        )
    return pd.DataFrame(rows)


def registry_players(save, season, artifacts) -> pd.DataFrame:
    from model import data

    players = data.read_output(save, season, "player-data").drop_duplicates("ID").set_index("ID")
    columns = registry.required_rating_columns(artifacts)
    return players[columns].apply(pd.to_numeric, errors="coerce")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--old", default="Model-Sim-26-Std", help="save of the older version (default OOTP 26)")
    parser.add_argument("--new", default="Model-Sim", help="save of the newer version (default OOTP 27)")
    parser.add_argument(
        "--ratings-from", default="Model-Sim", help="save whose ratings are scored with both model sets"
    )
    parser.add_argument("--season", type=int, default=2060, help="season of those ratings")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", default="docs/ootp26-vs-ootp27-attribute-impact.md")
    args = parser.parse_args(argv)

    sections, overall = [], []
    for name in ORDER:
        old_loaded, old = impact(args.old, name, args.folds, args.repeats)
        new_loaded, new = impact(args.new, name, args.folds, args.repeats)
        _, new_other = impact(args.new, name, args.folds, args.repeats, seed=1)
        merged = merge_impacts(old, new, seed_noise(new, new_other))
        merged["abs_step"] = merged["step_change"].abs()
        merged = merged.sort_values("abs_step", ascending=False).reset_index(drop=True)
        stream = new_loaded.notes["stream"]
        title = title_of(new_loaded)
        cv_old = old_loaded.metrics.get("cv", {}).get("r2")
        cv_new = new_loaded.metrics.get("cv", {}).get("r2")
        sections += [
            f"## {title}",
            "",
            f"Effects in **{UNITS[stream]}**. Model fit (player-grouped cross-validated R²): "
            f"{args.old} {fmt(cv_old, 3, False)}, {args.new} {fmt(cv_new, 3, False)}.",
            "",
            "| Rank | Attribute | +10 points, old | +10 points, new | Change | Change % | Share, old | Share, new | "
            "+1 SD, old | +1 SD, new | +1 SD change | Seed noise |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for rank, row in merged.iterrows():
            sections.append(
                f"| {rank + 1} | {label(row['feature'])} | {fmt(row['effect_step_old'])} | {fmt(row['effect_step_new'])} | "
                f"{fmt(row['step_change'])} | {ratio(row['effect_step_new'], row['effect_step_old'])} | "
                f"{pct(row['share_old'])} | {pct(row['share_new'])} | {fmt(row['effect_sd_old'])} | "
                f"{fmt(row['effect_sd_new'])} | {fmt(row['sd_change'])} | {fmt(row['sd_noise'], 2, False)} |"
            )
            overall.append(
                {
                    "model": title,
                    "attribute": label(row["feature"]),
                    "sd_old": row["effect_sd_old"],
                    "sd_new": row["effect_sd_new"],
                    "sd_change": row["sd_change"],
                    "noise": row["sd_noise"],
                }
            )
        sections.append("")
    top = pd.DataFrame(overall)
    top["abs"] = top["sd_change"].abs()
    top = top.sort_values("abs", ascending=False).head(25).reset_index(drop=True)

    same = same_ratings_comparison(args.old, args.new, args.ratings_from, args.season)
    head = [
        f"# Attribute impact: {args.old} (OOTP 26) against {args.new} (OOTP 27)",
        "",
        f"Generated {date.today()} by `scripts/compare_versions.py`. Both sets of models are trained on seasons 2046-2086 "
        "of their save; the effects are computed the same way for both (see `docs/model-attribute-ranking.md` for the method).",
        "",
        "## How to read it",
        "",
        "- **+10 points** is the average change of the predicted value when the rating goes up by 10 points; the **Change** "
        "is the 27 effect minus the 26 effect, so a positive change means the rating is worth more in OOTP 27.",
        "- **Share** is the permutation-importance share of the model's skill; a rating that moves together with another "
        "shares credit with it, so shares can shift between related ratings without a real change in value.",
        "- **+1 SD** uses one standard deviation of the rating in that save's training data and is the fair way to compare "
        "ratings with different spreads.",
        "- **Seed noise** is how much the +1 SD effect of the OOTP 27 model moves when only the cross-validation folds and "
        "scrambles change (seed 0 against seed 1). It is a floor: the two simulated leagues also contain different players "
        "and seasons, so the true uncertainty of a change is larger. Read changes of about 10-15 % or less, or of a few "
        "tenths of a run, as no real difference; the large ones (a rating whose effect doubles or halves) are the findings.",
        "- Units: batting and baserunning runs per 600 PA, pitching runs per 200 IP, fielding runs per season at the position.",
        "",
        "## Same ratings, different value",
        "",
        f"The {args.ratings_from} {args.season} ratings scored with both sets of models; the means are weighted by the "
        "playing time of that season. `Difference` is the OOTP 27 value minus the OOTP 26 value for the same ratings, in "
        "the stream's units. Both are values against the average of the league each model set was trained on, so a mean "
        "difference mostly says how good that league's average player is (the OOTP 27 league has better defenders, so the "
        "OOTP 26 models see its fielders as above average), not that a player is worth more in one version. Compare the "
        "spread (SD) and the rank correlation instead: the correlation says whether the two model sets order the same "
        "players the same way, the SD ratio how strongly they separate good from bad.",
        "",
        "| Value | Players | Mean, 26 models | Mean, 27 models | Difference | SD, 26 | SD, 27 | Rank correlation |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in same.iterrows():
        head.append(
            f"| {row['column']} | {row['players']} | {fmt(row['mean_old'])} | {fmt(row['mean_new'])} | "
            f"{fmt(row['difference'])} | {row['sd_old']:.2f} | {row['sd_new']:.2f} | {row['rank_corr']:.2f} |"
        )
    head += [
        "",
        "## Summary: the 25 largest changes of the effect of one standard deviation of a rating",
        "",
        "| Rank | Model | Attribute | +1 SD, 26 | +1 SD, 27 | Change | Seed noise |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for rank, row in top.iterrows():
        head.append(
            f"| {rank + 1} | {row['model']} | {row['attribute']} | {fmt(row['sd_old'])} | {fmt(row['sd_new'])} | "
            f"{fmt(row['sd_change'])} | {fmt(row['noise'], 2, False)} |"
        )
    head.append("")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(head + sections))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

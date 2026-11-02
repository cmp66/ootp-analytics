"""Rank the ratings used by the value models and write docs/model-attribute-ranking.md.

For every shipped model (batting and the two pitching parts as stacked platoon models, baserunning, one model per
fielding position) the ratings are ranked on held-out players: the model is refitted on player-grouped folds, and on each
held-out fold a rating is scrambled (permutation importance) and the loss in weighted R^2 is recorded. The effect of a
rating on the predicted value is measured as the mean change of the prediction when the rating is raised by 10 points and
by one standard deviation of the rating, over the held-out rows.

Usage: PYTHONPATH=. python scripts/rating_importance.py [save] [--folds 5] [--repeats 3] [--out docs/model-attribute-ranking.md]
       [--observations file.md]   (hand-written section; the built-in one describes the Model-Sim run)
"""

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from model import artifact, domains, evaluate, registry  # noqa: E402
from model.metrics import weighted_r2  # noqa: E402
from model.train import fit_table, make_model  # noqa: E402
from util.config import get_save  # noqa: E402

FLAGS = {"side_left", "bats_left", "bats_switch", "throws_left"}  # context for the model, not ratings
UNITS = {
    domains.BATTING: "runs per 600 PA",
    domains.BASERUNNING: "runs per 600 PA",
    domains.PITCHING_FIP: "runs per 200 IP",
    domains.PITCHING_BABIP: "runs saved per 200 IP (a lower BABIP against is better, so a positive effect is good)",
    domains.FIELDING: "runs per season (1200 innings)",
}
LABELS = {
    "BA": "BABIP (hit rating)",
    "GAP": "Gap power",
    "POW": "Power",
    "EYE": "Eye (walks)",
    "K": "Avoid strikeouts (AVK)",
    "STU": "Stuff",
    "CON": "Control",
    "PBABIP": "PBABIP",
    "HRR": "Home runs allowed (HRA)",
    "STM": "Stamina",
    "PIT": "Number of pitches (count)",
    "SPE": "Speed",
    "STE": "Stealing",
    "RUN": "Baserunning",
    "IF RNG": "Infield range",
    "IF ARM": "Infield arm",
    "IF ERR": "Infield error",
    "OF RNG": "Outfield range",
    "OF ARM": "Outfield arm",
    "C FRM": "Catcher framing",
    "C ABI": "Catcher ability (blocking)",
    "HT": "Height (cm)",
}
STEP = {"PIT": 1.0, "HT": 10.0}  # units of the "+10 points" column for features that are not 20-80 ratings
POSITIONS = {2: "C", 3: "1B", 4: "2B", 5: "3B", 6: "SS", 7: "LF", 8: "CF", 9: "RF"}

OBSERVATIONS = [
    "## What the ranking says (written by hand for the Model-Sim run of 2026-11-02; check it after retraining)",
    "",
    "- **Batting:** power (about half of the model's skill) and avoiding strikeouts (about a third) carry most of the value;",
    "  BABIP and eye are minor and gap power adds almost nothing once the others are known. +10 points of power is worth",
    "  about 9.6 runs per 600 PA, strikeout avoidance 6.7, BABIP 5.4, eye 3.5 and gap 1.8.",
    "- **Baserunning:** baserunning ability (`RUN`) and stealing (`STE`) are 92 % of it; raw speed adds little once those",
    "  are known (+10 points: 0.9, 1.1 and 0.3 runs per 600 PA).",
    "- **Pitching:** home runs allowed (49 %) leads, then stuff (28 %) and control (15 %): +10 points are worth 6.5, 4.9 and",
    "  3.3 runs per 200 IP. PBABIP acts through the BABIP part, where +10 points is worth about 2.7 runs per 200 IP.",
    "  Stamina and pitch count mostly mark the pitcher's role (starters average 55 stamina, relievers 38) and the value is",
    "  measured against his own usage group, so their small negative effects are role corrections, not penalties: among",
    "  starters they have no effect, among relievers a higher stamina is slightly worse (-0.17 runs per 200 IP per point).",
    "- **Fielding:** range is nearly everything at the outfield positions, 2B and 1B (75 % to 100 %). Infield arm matters most",
    "  at third base (59 % against 37 % for range), then shortstop (22 %) and little at second (5 %); outfield arm is worth",
    "  almost nothing (under 2 %); error rating 2 % to 4 % for infielders. Catcher framing is 96 % of the catcher's value.",
    "  +10 points of range: CF 13.6, SS 11.2, 2B 10.6, 3B 5.4, RF 4.8, LF 2.8, 1B 1.9 runs per season. First basemen get",
    "  13 % of their model from height.",
    "",
]


def importance_for(loaded, save: str, folds: int, repeats: int, seed: int = 0) -> pd.DataFrame:
    """Per rating: R^2 drop when scrambled, effect of +10 points and of +1 SD on the prediction (held-out rows)."""
    spec = registry.Spec(
        loaded.name, loaded.notes["stream"], loaded.notes["kind"], loaded.family, loaded.notes["position"]
    )
    table = registry.build_table(spec, save, loaded.seasons)
    frame = table.frame
    fold_of_player = evaluate.player_folds(frame[table.groups], folds, seed)
    fold_of_row = frame[table.groups].map(fold_of_player).to_numpy()
    ratings = [f for f in table.features if f not in FLAGS]
    drops = {f: [] for f in ratings}
    effects = {f: [] for f in ratings}
    sd_effects = {f: [] for f in ratings}
    rng = np.random.default_rng(seed)
    for fold in range(folds):
        train, test = np.flatnonzero(fold_of_row != fold), np.flatnonzero(fold_of_row == fold)
        model = fit_table(make_model(loaded.family, **loaded.params), table, train, seed)
        held = frame.iloc[test]
        x = held[table.features]
        base = model.predict(x)
        base_r2 = weighted_r2(held[table.target], base, held[table.weight])
        for feature in ratings:
            for _ in range(repeats):
                scrambled = x.copy()
                scrambled[feature] = rng.permutation(scrambled[feature].to_numpy())
                drops[feature].append(
                    base_r2 - weighted_r2(held[table.target], model.predict(scrambled), held[table.weight])
                )
            sd = float(frame[feature].std())
            for step, store in ((STEP.get(feature, 10.0), effects), (sd, sd_effects)):
                raised = x.copy()
                raised[feature] = raised[feature] + step
                store[feature].append(float(np.average(model.predict(raised) - base, weights=held[table.weight])))
    rows = [
        {
            "feature": f,
            "r2_drop": float(np.mean(drops[f])),
            "effect_step": float(np.mean(effects[f])),
            "effect_sd": float(np.mean(sd_effects[f])),
            "sd": float(frame[f].std()),
        }
        for f in ratings
    ]
    out = pd.DataFrame(rows)
    positive = out["r2_drop"].clip(lower=0)
    out["share"] = positive / positive.sum() if positive.sum() > 0 else np.nan
    return out.sort_values("r2_drop", ascending=False).reset_index(drop=True)


def format_table(df: pd.DataFrame, divisor: float = 1.0, digits: int = 2) -> list[str]:
    lines = [
        "| Rank | Attribute | Share of the model's skill | R² lost when scrambled | Effect of +10 points "
        "| Effect of +1 SD (SD) |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for rank, row in df.iterrows():
        label = LABELS.get(row["feature"], row["feature"])
        step = "+1 pitch" if row["feature"] == "PIT" else ("+10 cm" if row["feature"] == "HT" else "+10 points")
        share = "n/a" if pd.isna(row["share"]) else f"{row['share']:.0%}"
        lines.append(
            f"| {rank + 1} | {label} (`{row['feature']}`) | {share} | {row['r2_drop']:.3f} | "
            f"{row['effect_step'] / divisor:+.{digits}f} ({step}) | {row['effect_sd'] / divisor:+.{digits}f} ({row['sd']:.1f}) |"
        )
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("save", nargs="?", default="Model-Sim")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", default="docs/model-attribute-ranking.md")
    parser.add_argument("--observations", help="markdown file with the hand-written observations of this save")
    args = parser.parse_args()
    ratings_set = get_save(args.save).ratings
    folder = artifact.models_dir(args.save, ratings_set)
    shipped = {}
    for path in sorted(folder.glob("*.joblib")):
        loaded = artifact.load_artifact(args.save, ratings_set, path.stem)
        primary = loaded.notes["kind"] == "split" or loaded.notes["stream"] in (domains.BASERUNNING, domains.FIELDING)
        if primary:
            shipped[loaded.name] = loaded
    if not shipped:
        print(f"no trained models in {folder}; run scripts/train_models.py {args.save}", file=sys.stderr)
        return 2

    sections, summary = [], []
    order = ["batting-split", "baserunning", "pitching-fip-split", "pitching-babip-split"] + [
        f"fielding-{p}" for p in domains.FIELD_POSITIONS
    ]
    for name in [n for n in order if n in shipped]:
        loaded = shipped[name]
        table = importance_for(loaded, args.save, args.folds, args.repeats)
        stream = loaded.notes["stream"]
        title = {
            "batting-split": "Batting",
            "baserunning": "Baserunning",
            "pitching-fip-split": "Pitching: FIP part (strikeouts, walks, home runs)",
            "pitching-babip-split": "Pitching: BABIP part (hits on balls in play)",
        }.get(name, f"Fielding at {POSITIONS.get(loaded.notes['position'], '?')}")
        cv = loaded.metrics.get("cv", {})
        sections += [
            f"## {title}",
            "",
            f"Model `{name}` ({loaded.family}), {loaded.metrics.get('rows', '?')} rows, trained on seasons "
            f"{loaded.seasons[0]}-{loaded.seasons[-1]}"
            + (f", player-grouped cross-validated R² {cv['r2']:.3f}." if cv else "."),
            f"Effects are in **{UNITS[stream]}**.",
            "",
        ]
        divisor = 1.0
        if stream == domains.PITCHING_BABIP:
            conversion = loaded.notes["conversion"]
            per_babip = conversion["bip_per_200_ip"] * conversion["hit_value"]
            divisor = -1.0 / per_babip  # BABIP difference -> runs saved per 200 IP
            sections += [
                "This model explains almost none of the variance (almost all of a pitcher's BABIP is luck), so its ranking is "
                "not informative; the effect of the rating is. A BABIP difference of 0.001 over 200 innings is about "
                f"{per_babip / 1000:.2f} runs ({conversion['bip_per_200_ip']:.0f} balls in play x {conversion['hit_value']:.2f} "
                "runs per hit).",
                "",
            ]
        sections += format_table(table, divisor=divisor, digits=2) + [""]
        for _, row in table.iterrows():
            summary.append(
                {
                    "stream": title,
                    "attribute": f"{LABELS.get(row['feature'], row['feature'])} (`{row['feature']}`)",
                    "effect_sd": row["effect_sd"] / divisor,
                    "share": np.nan if stream == domains.PITCHING_BABIP else row["share"],
                }
            )
    top = pd.DataFrame(summary)
    top["abs"] = top["effect_sd"].abs()
    top = top.sort_values("abs", ascending=False).head(20).reset_index(drop=True)
    head = [
        "# Which ratings matter: ranking of the model attributes",
        "",
        f"Generated {date.today()} by `scripts/rating_importance.py` from the models trained on **{args.save}** "
        f"(ratings set `{ratings_set}`). Regenerate after retraining.",
        "",
        "## How to read it",
        "",
        "- **Share of the model's skill** is the loss in held-out R² when the rating is scrambled (permutation importance on",
        "  players the model has not seen, five player-grouped folds), as a share of the total over the ratings of the model.",
        "  Ratings that move together (for example range and the composite position rating) share their credit.",
        "- **Effect of +10 points** is the average change of the predicted value when the rating goes up by 10 points (two",
        "  steps on the 20-80 scale) with everything else unchanged. **Effect of +1 SD** raises the rating by one standard",
        "  deviation of that rating among the players of the model (the SD is in brackets), which makes ratings with a",
        "  different spread comparable. Where a model is non-linear (boosting, network) the effect is an average.",
        "- 'R² lost when scrambled' can exceed 1: a scrambled rating that the model relies on gives predictions worse than",
        "  the average, so R² goes negative.",
        "- Units differ by stream: batting and baserunning are runs per 600 PA, pitching runs per 200 IP, fielding runs per",
        "  full season at the position. A regular's 600 PA, a starter's 200 IP and a fielder's 1200 innings are each about a",
        "  season, so the summary table compares the +1 SD effects across streams as seasonal runs; a reliever throws about",
        "  a third of 200 IP, so his seasonal effects are about a third of the pitching figures.",
        "- Batting and pitching come from the stacked platoon models: BABIP (`BA`), gap power, power, eye and AVK are the",
        "  ratings against the pitcher (batter) of the row, vs right and vs left combined in one attribute.",
        "- `CON` and `MOV` are derived ratings (see `docs/derived-ratings.md`) and are not in the models; contact is half",
        "  BABIP and half AVK, movement mostly home runs allowed.",
        f"- The ranking is for the {ratings_set} rating scale of this save; a save with a different scale needs its own",
        "  models and its own ranking. Ratings that are not listed for a stream had no measurable value in the attribute review",
        "  (`plans/14-models.md`).",
        "",
        "## Summary: the 20 biggest effects of one standard deviation of a rating on seasonal value",
        "",
        "| Rank | Stream | Attribute | Effect of +1 SD (runs per season) | Share within its model |",
        "|---:|---|---|---:|---:|",
    ]
    for rank, row in top.iterrows():
        share = "n/a" if pd.isna(row["share"]) else f"{row['share']:.0%}"
        head.append(f"| {rank + 1} | {row['stream']} | {row['attribute']} | {row['effect_sd']:+.2f} | {share} |")
    head.append("")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.observations:
        observations = Path(args.observations).read_text().rstrip("\n").split("\n") + [""]
    elif args.save == "Model-Sim":
        observations = OBSERVATIONS
    else:
        observations = []
    out.write_text("\n".join(head + observations + sections))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

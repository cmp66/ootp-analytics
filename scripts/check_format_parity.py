"""Check that what the models need is in both the dump (training leagues) and the export (human leagues), and how it converts.

Usage:
    PYTHONPATH=. python scripts/check_format_parity.py [--out docs/export-dump-parity.md]

For each ratings set that has a dump save and an export save (config/saves.toml) it compares the processed player-data of
the latest season of each: the rating columns the models read, the traits and proneness (numbers against categories and
the cut points that convert one into the other), the level, how a draftee's route is told without the college flag, and
the fields only one format has. Run it after changing the import or the conversions.
"""

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from model import artifact, data, formats, outlook_data, registry  # noqa: E402
from util.config import load_saves  # noqa: E402
from util.seasons import available_seasons  # noqa: E402

DUMP_ONLY = {
    "college": "Export: no college flag (HSC is empty). Convert: age at the draft >= 20 means college (checked below).",
    "acquired, free_agent, retired, league_id, last_team_id": (
        "Export: Lev (level), TXN (last transaction), ORG / TM (team). Level converts to a group (below)."
    ),
    "draft_year, draft_round, draft_overall_pick, picked_in_draft": (
        "Export: Draft, Round, Pick, OA Pick (0 = not drafted). Converts directly."
    ),
    "prone_leg, prone_back, prone_arm": "Not in the export. Models for human leagues use only overall proneness.",
    "injury_is_injured, injury_left, injury_career_ending": "Not in the export player list. Not needed to score a player.",
    "experience": "Different from the export's MLY (MLB years). Not used as a feature.",
    "personality_*": (
        "Converted to WE, INT, LEA, LOY, FIN in the player-data of both formats (numbers 1-200 in the dump, 40 / 100 / 160 "
        "codes in the export; use the `<trait> cat` categories)."
    ),
}
EXPORT_ONLY = {
    "AD (adaptability)": "In no dump file: it cannot be modelled; the viewer can show it from the export.",
    "Lev, LG, TXN, Draft, Round, Pick, OA Pick, DOB, OVR, POT": (
        "Used for level and entry fields; OVR / POT are the game's star ratings."
    ),
    "MLY, MLD, YL, ROOKIE, R5, COMP, Type, PT, SLR, ...": (
        "Service time, rookie status, personality type and contract fields; not used by the models."
    ),
}


def latest_output(save: str) -> tuple[int, pd.DataFrame]:
    season = available_seasons(save)[-1]
    return season, data.read_output(save, season, "player-data")


def numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def needed_rating_columns(models_from: str, ratings: str) -> list[str]:
    folder = artifact.models_dir(models_from, ratings)
    artifacts = {p.stem: artifact.load_artifact(models_from, ratings, p.stem) for p in sorted(folder.glob("*.joblib"))}
    columns = registry.required_rating_columns(artifacts) if artifacts else []
    extra = [*outlook_data.GATE_RATINGS, *outlook_data.POSITION_RATINGS, *outlook_data.POTENTIAL_RATINGS]
    return list(dict.fromkeys([*columns, *extra, *formats.TRAITS, formats.PRONE, "Age", "HT", "WT", "B", "T"]))


def rating_table(dump: pd.DataFrame, export: pd.DataFrame, columns: list[str]) -> tuple[list[str], list[str]]:
    lines = [
        "| Column | In dump | In export | Dump min-max | Export min-max | On the 5-point grid, dump / export |",
        "|---|---|---|---|---|---:|",
    ]
    problems = []
    for column in columns:
        in_dump, in_export = column in dump.columns, column in export.columns
        if not (in_dump and in_export):
            problems.append(column)
            lines.append(f"| {column} | {'yes' if in_dump else '**no**'} | {'yes' if in_export else '**no**'} | | | |")
            continue
        d, e = numeric(dump, column).dropna(), numeric(export, column).dropna()
        if d.empty or e.empty:
            lines.append(f"| {column} | yes | yes | text | text | |")
            continue
        grid = lambda v: f"{(v % 5 == 0).mean():.0%}"  # noqa: E731
        lines.append(
            f"| {column} | yes | yes | {d.min():g}-{d.max():g} | {e.min():g}-{e.max():g} | {grid(d)} / {grid(e)} |"
        )
    return lines, problems


def trait_section(ratings: str, dump: pd.DataFrame, export: pd.DataFrame) -> list[str]:
    lines = [
        "| Trait | Export shares (labels low to high) | Cut points on the dump numbers | Dump mean of each category |",
        "|---|---|---|---|",
    ]
    bounds = outlook_data.load_trait_bounds(ratings)
    for trait in (*formats.TRAITS, formats.PRONE):
        labels = outlook_data.trait_labels(trait)
        raw = export[f"{trait} cat"]
        categories = raw.map(formats.PRONE3) if trait == formats.PRONE else raw
        shares = formats.label_shares(categories, labels)
        text = ", ".join(f"{label} {share:.1%}" for label, share in shares.items())
        if bounds and trait in bounds:
            values = numeric(dump, trait)
            category = formats.categorize(values, bounds[trait], labels)
            means = formats.category_means(values, category)
            lines.append(
                f"| {trait} | {text} | {', '.join(f'{b:g}' for b in bounds[trait])} | "
                + ", ".join(f"{label} {means.get(label, float('nan')):.0f}" for label in labels)
                + " |"
            )
        else:
            lines.append(f"| {trait} | {text} | not fitted (run scripts/build_outlook_tables.py) | |")
    return lines


def level_section(dump_save: str, dump: pd.DataFrame, export: pd.DataFrame, season: int) -> list[str]:
    levels = outlook_data.league_levels(dump_save, season)
    free = numeric(dump, "free_agent") == 1
    dump_groups = pd.Series(
        [formats.level_group(v, "dump", fa) for v, fa in zip(dump["league_id"].map(levels), free)], index=dump.index
    )
    export_groups = export["Lev"].map(lambda v: formats.level_group(v, "export"))
    table = pd.concat(
        [
            dump_groups.value_counts(normalize=True).rename("dump"),
            export_groups.value_counts(normalize=True).rename("export"),
        ],
        axis=1,
    ).fillna(0)
    lines = ["| Level group | Dump share | Export share |", "|---|---:|---:|"]
    lines += [f"| {group} | {row['dump']:.1%} | {row['export']:.1%} |" for group, row in table.iterrows()]
    return lines


def draft_section(dump_save: str) -> list[str]:
    """How well the age at the draft finds the college draftees of the dump (the export has no college flag).

    The dump has one college flag for junior college and four-year players, so they are one group here too."""
    rows = []
    for season in available_seasons(dump_save)[::6]:
        frame = data.read_output(dump_save, season, "player-data")
        drafted = frame[(numeric(frame, "picked_in_draft") == 1) & (numeric(frame, "draft_year") > 0)]
        age_at_draft = numeric(drafted, "Age") - (season - numeric(drafted, "draft_year"))
        rows.append(pd.DataFrame({"age": age_at_draft, "college": numeric(drafted, "college") == 1}))
    both = pd.concat(rows)
    predicted = both["age"] >= formats.AGE_AT_DRAFT_COLLEGE
    accuracy = float((both["college"] == predicted).mean())
    by_age = both[both["age"].between(17, 23)].groupby("age")["college"].agg(["size", "mean"])
    lines = [
        f"Drafted players of {len(rows)} sample seasons of {dump_save}: {len(both)}. The rule 'college from age "
        f"{formats.AGE_AT_DRAFT_COLLEGE}' agrees with the college flag for {accuracy:.1%} of them. (Rule from age 19: "
        f"{float((both['college'] == (both['age'] >= 19)).mean()):.1%}.)",
        "",
        "| Age at the draft | " + " | ".join(f"{int(a)}" for a in by_age.index) + " |",
        "|---|" + "---:|" * len(by_age),
        "| Players | " + " | ".join(f"{int(n)}" for n in by_age["size"]) + " |",
        "| College flag | " + " | ".join(f"{m:.1%}" for m in by_age["mean"]) + " |",
    ]
    return lines


def pair_report(ratings: str, dump_save: str, export_save: str) -> tuple[list[str], list[str]]:
    dump_season, dump = latest_output(dump_save)
    export_season, export = latest_output(export_save)
    columns = needed_rating_columns(dump_save, ratings)
    table, problems = rating_table(dump, export, columns)
    lines = [
        f"## Ratings set {ratings}: {dump_save} {dump_season} (dump) against {export_save} {export_season} (export)",
        "",
        f"Dump player-data {dump.shape[0]} rows, export {export.shape[0]} rows.",
        "",
        "### Rating and context columns the models read",
        "",
        *table,
        "",
        "### Traits and proneness: numbers in the dump, categories in the export",
        "",
        "Higher proneness is more injury prone (it goes with less playing time). Iron Man (sustained 20 or less) and Wrecked "
        "(sustained 180 or more) depend on history an export does not carry, so proneness is used in three levels "
        "(Durable = Iron Man + Durable, Normal, Fragile = Fragile + Wrecked).",
        "",
        *trait_section(ratings, dump, export),
        "",
        "### Level",
        "",
        *level_section(dump_save, dump, export, dump_season),
        "",
        "### Draft route",
        "",
        *draft_section(dump_save),
        "",
    ]
    return lines, problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", default="docs/export-dump-parity.md")
    args = parser.parse_args(argv)
    saves = load_saves()
    sections, all_problems = [], {}
    for ratings in sorted({s.ratings for s in saves.values()}):
        dumps = [n for n, s in saves.items() if s.ratings == ratings and s.source == "dump" and available_seasons(n)]
        exports = [
            n for n, s in saves.items() if s.ratings == ratings and s.source == "export" and available_seasons(n)
        ]
        if not dumps or not exports:
            continue
        lines, problems = pair_report(ratings, dumps[0], exports[0])
        sections += lines
        all_problems[ratings] = problems
    head = [
        "# Dump and export: what the models need and how it converts",
        "",
        f"Generated {date.today()} by `scripts/check_format_parity.py`. The long training leagues exist only as database "
        "dumps; the "
        "human leagues only as in-game exports (a dump needs commissioner rights). A model that scores both must use only "
        "what both hold.",
        "",
        "## Summary",
        "",
        *[
            f"- {ratings}: "
            + (
                "every column the models read is in both formats."
                if not p
                else f"MISSING in one format: {', '.join(p)}."
            )
            for ratings, p in all_problems.items()
        ],
        "",
        "## Only in the dump",
        "",
        "| Field | Export equivalent / conversion |",
        "|---|---|",
        *[f"| {k} | {v} |" for k, v in DUMP_ONLY.items()],
        "",
        "## Only in the export",
        "",
        "| Field | Note |",
        "|---|---|",
        *[f"| {k} | {v} |" for k, v in EXPORT_ONLY.items()],
        "",
        "## Not available in the export at all",
        "",
        "- Adaptability in the dump (so no development effect can be measured).",
        "- Playing time of earlier seasons (a human save has few seasons; the lagged playing-time features need two or more).",
        "- Entry route of undrafted players: international signings cannot be told from other undrafted entries once "
        "they are not new.",
        "",
    ]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(head + sections))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

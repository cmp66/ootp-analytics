"""Build the team-season table used to study whether pitcher/batter BABIP ratings move BIZ chances.

Analysis only (plans/10-biz-babip-effect.md); not part of the pipeline.

Usage: python scripts/analyze_biz_babip.py [save] [ratings] [out_csv]
Defaults: Model-Sim Standard27 plans/data/biz-team-seasons.csv
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TIERS = ["R", "L", "E", "U", "Z"]  # difficulty tiers; I (impossible) is never made and left out
IF_POS, OF_POS = [3, 4, 5, 6], [7, 8, 9]


def weighted(df: pd.DataFrame, value: str, weight: str, by: str = "team_id") -> pd.Series:
    w = df[weight]
    return (df[value] * w).groupby(df[by]).sum() / w.groupby(df[by]).sum()


def season_table(folder: Path, season: int) -> pd.DataFrame:
    def read(name: str) -> pd.DataFrame:
        return pd.read_csv(folder / f"{name}.csv", low_memory=False)

    def major(df: pd.DataFrame, split: int | None = None) -> pd.DataFrame:
        df = df[(df["year"] == season) & (df["level_id"] == 1)]
        return df if split is None else df[df["split_id"] == split]

    fielding = major(read("players_career_fielding_stats"))
    ratings_f = read("players_fielding")[
        ["player_id", "fielding_ratings_infield_range", "fielding_ratings_outfield_range"]
    ]
    fielding = fielding.merge(ratings_f, on="player_id", how="left")
    # innings with the extra outs, so the weights match the import
    fielding["ip_clean"] = fielding["ip"] + fielding["ipf"] / 3

    teams = pd.DataFrame(index=pd.Index(sorted(fielding["team_id"].unique()), name="team_id"))
    teams["season"] = season
    teams["ip"] = fielding.groupby("team_id")["ip_clean"].sum() / 9
    for i, tier in enumerate(TIERS):
        teams[f"att_{tier}"] = fielding.groupby("team_id")[f"opps_{i}"].sum()
        teams[f"made_{tier}"] = fielding.groupby("team_id")[f"opps_made_{i}"].sum()
    teams["att_I"] = fielding.groupby("team_id")["opps_5"].sum()
    teams["att"] = teams[[f"att_{t}" for t in TIERS]].sum(axis=1)
    teams["made"] = teams[[f"made_{t}" for t in TIERS]].sum(axis=1)

    # attempts per team by position group, for the ground-ball / fly-ball split
    for name, positions in [("if", IF_POS), ("of", OF_POS), ("p", [1]), ("c", [2])]:
        sel = fielding[fielding["position"].isin(positions)]
        teams[f"att_{name}"] = sel.groupby("team_id")[[f"opps_{i}" for i in range(5)]].sum().sum(axis=1)
        teams[f"made_{name}"] = sel.groupby("team_id")[[f"opps_made_{i}" for i in range(5)]].sum().sum(axis=1)

    # fielders' own range, innings-weighted, as a skill control
    inf = fielding[fielding["position"].isin(IF_POS)]
    out = fielding[fielding["position"].isin(OF_POS)]
    teams["if_range"] = weighted(inf, "fielding_ratings_infield_range", "ip_clean")
    teams["of_range"] = weighted(out, "fielding_ratings_outfield_range", "ip_clean")

    pitching = major(read("players_career_pitching_stats"), split=1)
    pr = read("players_pitching")[["player_id", "pitching_ratings_overall_pbabip", "pitching_ratings_misc_ground_fly"]]
    pitching = pitching.merge(pr, on="player_id", how="left")
    teams["pbabip"] = weighted(pitching, "pitching_ratings_overall_pbabip", "bf")
    teams["gb_rating"] = weighted(pitching, "pitching_ratings_misc_ground_fly", "bf")
    lg_pbabip = (pitching["pitching_ratings_overall_pbabip"] * pitching["bf"]).sum() / pitching["bf"].sum()
    teams["lg_pbabip"] = lg_pbabip

    batting = major(read("players_career_batting_stats"), split=1)
    br = read("players_batting")[["player_id", "batting_ratings_overall_babip"]]
    batting = batting.merge(br, on="player_id", how="left")
    teams["babip_rating"] = weighted(batting, "batting_ratings_overall_babip", "pa")  # own hitters (control)
    teams["lg_babip_rating"] = (batting["batting_ratings_overall_babip"] * batting["pa"]).sum() / batting["pa"].sum()

    tp = major(read("team_pitching_stats"), split=1).set_index("team_id")
    bip = tp["ab"] - tp["k"] - tp["hra"] + tp["sf"]
    teams["babip_against"] = (tp["ha"] - tp["hra"]) / bip
    teams["bip"] = bip
    teams["bf"] = tp["bf"]
    teams["k"] = tp["k"]
    teams["bb"] = tp["bb"]
    teams["gb"] = tp["gb"]
    teams["fb"] = tp["fb"]
    return teams.reset_index()


def main() -> None:
    save = sys.argv[1] if len(sys.argv) > 1 else "Model-Sim"
    ratings = sys.argv[2] if len(sys.argv) > 2 else "Standard27"
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else ROOT / "plans" / "data" / "biz-team-seasons.csv"
    base = ROOT / "files" / save / "dumps" / ratings
    frames = []
    for folder in sorted(base.iterdir()):
        if folder.is_dir() and folder.name.isdigit():
            frames.append(season_table(folder, int(folder.name)))
            print(folder.name, len(frames[-1]), "teams")
    table = pd.concat(frames, ignore_index=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    print(f"wrote {out} ({len(table)} team-seasons)")


if __name__ == "__main__":
    main()

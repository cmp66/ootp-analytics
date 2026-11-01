import pandas as pd
from util import paths
from util.innings import dump_ip_to_outs, outs_to_ip
from util.config import get_save
import os
from model import (
    convert_personalilty_trait,
    convert_height_to_cm,
    convert_groundball_flyball,
    convert_velocity,
    convert_throws,
    convert_pitch_type,
    convert_slot,
    convert_bat_rl,
    convert_gbt,
    convert_fbt,
    convert_dump_positions,
    convert_stars,
    convert_injury_prone,
)

# Column that uniquely identifies each kind of in-game export file
EXPORT_FILE_MARKERS = {
    "team_fielding": "CERA",
    "team_pitching": "WIN%",
    "team_batting": "wRC",
    "player_batting": "PI/PA",
    "player_pitching": "IRS",
    "player_fielding": "BIZ-R",
}

STAT_COLUMN_RENAMES = {
    "player_batting": {
        "1B": "SINGLE",
        "2B": "DOUBLE",
        "3B": "TRIPLE",
        "BABIP": "BABIP-O",
        "K": "SO",
    },
    "player_pitching": {
        "1B": "SINGLE",
        "2B": "DOUBLE",
        "3B": "TRIPLE",
        "BABIP": "BABIP-O",
        "FB": "FLYB",
        "GB": "GROUNDB",
    },
}


def classify_export_files(filedir) -> dict[str, list[pd.DataFrame]]:
    """Read every csv in an export season folder and group them by table type."""
    tables: dict[str, list[pd.DataFrame]] = {name: [] for name in EXPORT_FILE_MARKERS}
    for file in sorted(
        f for f in os.listdir(filedir) if os.path.isfile(os.path.join(filedir, f))
    ):
        df_temp = pd.read_csv(f"{filedir}/{file}")
        for name, marker in EXPORT_FILE_MARKERS.items():
            if marker in df_temp.columns:
                tables[name].append(df_temp)
                break
    missing = [name for name, frames in tables.items() if not frames]
    if missing:
        raise FileNotFoundError(f"Export folder {filedir} has no file for: {missing}")
    return tables


def combine_export_tables(frames: list[pd.DataFrame], keys: list[str]) -> pd.DataFrame:
    """Stack the files of one table type (e.g. one fielding file per position) and check the keys are unique."""
    df = pd.concat(frames, ignore_index=True).drop_duplicates()
    duplicated = df.duplicated(keys)
    if duplicated.any():
        raise ValueError(
            f"{int(duplicated.sum())} rows share the same {keys}: {df.loc[duplicated, keys].head().to_dict('list')}"
        )
    return df


def import_csv_flex(
    league: str, season: int, basedir
) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:

    tables = classify_export_files(paths.export_dir(league, season))

    df_player_batting_stats = combine_export_tables(tables["player_batting"], ["ID"])
    df_player_pitching_stats = combine_export_tables(tables["player_pitching"], ["ID"])
    df_player_fielding_stats = combine_export_tables(
        tables["player_fielding"], ["ID", "POS"]
    )
    df_team_batting = combine_export_tables(tables["team_batting"], ["ID"])
    df_team_pitching = combine_export_tables(tables["team_pitching"], ["ID"])
    df_team_fielding = combine_export_tables(tables["team_fielding"], ["ID"])

    # Need to rename columns which have the same name in the two table (batting and ratings)
    df_player_batting_stats = df_player_batting_stats.rename(
        columns=STAT_COLUMN_RENAMES["player_batting"]
    )
    df_player_pitching_stats = df_player_pitching_stats.rename(
        columns=STAT_COLUMN_RENAMES["player_pitching"]
    )

    return (
        df_player_batting_stats,
        df_player_pitching_stats,
        df_player_fielding_stats,
        df_team_batting,
        df_team_pitching,
        df_team_fielding,
    )


# Export ratings column names that differ from the dump / duplicate between tables
EXPORT_RATINGS_RENAMES = {
    "ROOK": "ROOKIE",
    "R": "REGION",
    "CON vR.1": "CON.1 vR",
    "CON vR_1": "CON.1 vR",
    "CON vL.1": "CON.1 vL",
    "CON vL_1": "CON.1 vL",
    "CON P.1": "CON.1 P",
    "CON_1": "CON.1",
    "CON P_1": "CON.1 P",
    "HT P": "BA P",
    "POS": "LPOS",
    "HRA": "HRR",
    "HRA P": "HRR P",
    "HRA vL": "HRR vL",
    "HRA vR": "HRR vR",
}


def import_ratings(
    league: str, season: int, ratings_type: str, basedir: str
) -> pd.DataFrame:

    filedir = paths.export_ratings_dir(league, season)
    frames = [pd.read_csv(filedir / file) for file in sorted(os.listdir(filedir))]
    if not frames:
        raise FileNotFoundError(f"No ratings files in {filedir}")
    df_player_ratings = pd.concat(frames, ignore_index=True)

    df_player_ratings = df_player_ratings.rename(columns=EXPORT_RATINGS_RENAMES)
    df_player_ratings = df_player_ratings.drop(columns=["EXP"], errors="ignore")
    df_player_ratings.set_index("ID", inplace=True)

    df_player_ratings["LEA"] = df_player_ratings["LEA"].apply(
        convert_personalilty_trait
    )
    df_player_ratings["LOY"] = df_player_ratings["LOY"].apply(
        convert_personalilty_trait
    )
    df_player_ratings["AD"] = df_player_ratings["AD"].apply(convert_personalilty_trait)
    df_player_ratings["FIN"] = df_player_ratings["FIN"].apply(
        convert_personalilty_trait
    )
    df_player_ratings["WE"] = df_player_ratings["WE"].apply(convert_personalilty_trait)
    df_player_ratings["INT"] = df_player_ratings["INT"].apply(
        convert_personalilty_trait
    )
    df_player_ratings["Prone"] = df_player_ratings["Prone"].apply(convert_injury_prone)

    df_player_ratings["WT"] = df_player_ratings["WT"].apply(lambda x: int(x[:3]))
    df_player_ratings["HT"] = df_player_ratings["HT"].apply(convert_height_to_cm)

    df_player_ratings["G/F"] = df_player_ratings["G/F"].apply(
        convert_groundball_flyball
    )
    df_player_ratings["VELO"] = df_player_ratings["VELO"].apply(convert_velocity)
    df_player_ratings["VT"] = df_player_ratings["VT"].apply(convert_velocity)
    df_player_ratings["T"] = df_player_ratings["T"].apply(convert_throws)
    df_player_ratings["PT"] = df_player_ratings["PT"].apply(convert_pitch_type)
    df_player_ratings["Slot"] = df_player_ratings["Slot"].apply(convert_slot)

    # player_data["BBT"] = player_data["BBT"].apply(convert_bbt)
    df_player_ratings["GBT"] = df_player_ratings["GBT"].apply(convert_gbt)
    df_player_ratings["FBT"] = df_player_ratings["FBT"].apply(convert_fbt)
    df_player_ratings["B"] = df_player_ratings["B"].apply(convert_bat_rl)

    df_player_ratings["Lev"] = df_player_ratings["Lev"].apply(
        lambda x: "R" if x == "R-" else x
    )
    df_player_ratings["Lev"] = df_player_ratings["Lev"].apply(
        lambda x: "FA" if x in ["-", "", "0"] else x
    )
    df_player_ratings["ORG"] = df_player_ratings["ORG"].apply(
        lambda x: "FA" if x in ["-", "", "0"] else x
    )

    df_player_ratings["POT"] = df_player_ratings["POT"].apply(convert_stars)
    df_player_ratings["OVR"] = df_player_ratings["OVR"].apply(convert_stars)

    return df_player_ratings


def import_ootp_dump_ratings(
    league: str, season: int, ratings_type: str, basedir: str
) -> pd.DataFrame:
    filedir = paths.dump_dir(league, ratings_type, season)

    df_player_ratings_base = pd.read_csv(f"{filedir}/players.csv")
    df_player_batting_ratings_base = pd.read_csv(f"{filedir}/players_batting.csv")
    df_player_fielding_ratings_base = pd.read_csv(f"{filedir}/players_fielding.csv")
    df_player_pitching_ratings_base = pd.read_csv(f"{filedir}/players_pitching.csv")

    df_player_ratings_base.rename(
        columns={
            "player_id": "ID",
            "team_id": "ORG",
            "first_name": "First Name",
            "last_name": "Last Name",
            "position": "LPOS",
            "weight": "WT",
            "height": "HT",
            "bats": "B",
            "throws": "T",
            "age": "Age",
            "personality_greed": "FIN",
            "personality_loyalty": "LOY",
            "personality_work_ethic": "WE",
            "personality_intelligence": "INT",
            "personality_leader": "LEA",
            "prone_overall": "Prone",
        },
        inplace=True,
    )
    df_player_batting_ratings_base.rename(
        columns={
            "player_id": "ID",
            "team_id": "Team Name",
            "batting_ratings_overall_contact": "CON",
            "batting_ratings_overall_gap": "GAP",
            "batting_ratings_overall_eye": "EYE",
            "batting_ratings_overall_strikeouts": "K's",
            "batting_ratings_overall_power": "POW",
            "batting_ratings_overall_babip": "BABIP",
            "batting_ratings_vsr_contact": "CON vR",
            "batting_ratings_vsr_gap": "GAP vR",
            "batting_ratings_vsr_power": "POW vR",
            "batting_ratings_vsr_eye": "EYE vR",
            "batting_ratings_vsr_strikeouts": "K vR",
            "batting_ratings_vsr_babip": "BA vR",
            "batting_ratings_vsl_contact": "CON vL",
            "batting_ratings_vsl_gap": "GAP vL",
            "batting_ratings_vsl_power": "POW vL",
            "batting_ratings_vsl_eye": "EYE vL",
            "batting_ratings_vsl_strikeouts": "K vL",
            "batting_ratings_vsl_babip": "BA vL",
            "batting_ratings_talent_contact": "CON P",
            "batting_ratings_talent_gap": "GAP P",
            "batting_ratings_talent_power": "POW P",
            "batting_ratings_talent_eye": "EYE P",
            "batting_ratings_talent_strikeouts": "K P",
            "batting_ratings_talent_babip": "BA P",
            "batting_ratings_misc_bunt": "BUN",
            "batting_ratings_misc_bunt_for_hit": "BFH",
            "batting_ratings_misc_gb_hitter_type": "GBT",
            "batting_ratings_misc_fb_hitter_type": "FBT",
            "running_ratings_speed": "SPE",
            "running_ratings_stealing_rate": "SR",
            "running_ratings_stealing": "STE",
            "running_ratings_baserunning": "RUN",
        },
        inplace=True,
    )
    df_player_fielding_ratings_base.rename(
        columns={
            "player_id": "ID",
            "team_id": "Team Name",
            "fielding_ratings_infield_range": "IF RNG",
            "fielding_ratings_infield_error": "IF ERR",
            "fielding_ratings_infield_arm": "IF ARM",
            "fielding_ratings_turn_doubleplay": "TDP",
            "fielding_ratings_outfield_range": "OF RNG",
            "fielding_ratings_outfield_error": "OF ERR",
            "fielding_ratings_outfield_arm": "OF ARM",
            "fielding_ratings_catcher_arm": "C ARM",
            "fielding_ratings_catcher_ability": "C ABI",
            "fielding_ratings_catcher_framing": "C FRM",
            "fielding_rating_pos1": "P",
            "fielding_rating_pos2": "C",
            "fielding_rating_pos3": "1B",
            "fielding_rating_pos4": "2B",
            "fielding_rating_pos5": "3B",
            "fielding_rating_pos6": "SS",
            "fielding_rating_pos7": "LF",
            "fielding_rating_pos8": "CF",
            "fielding_rating_pos9": "RF",
            "fielding_rating_pos1_pot": "P Pot",
            "fielding_rating_pos2_pot": "C Pot",
            "fielding_rating_pos3_pot": "1B Pot",
            "fielding_rating_pos4_pot": "2B Pot",
            "fielding_rating_pos5_pot": "3B Pot",
            "fielding_rating_pos6_pot": "SS Pot",
            "fielding_rating_pos7_pot": "LF Pot",
            "fielding_rating_pos8_pot": "CF Pot",
            "fielding_rating_pos9_pot": "RF Pot",
        },
        inplace=True,
    )

    df_player_pitching_ratings_base.rename(
        columns={
            "player_id": "ID",
            "team_id": "Team Name",
            "pitching_ratings_overall_stuff": "STU",
            "pitching_ratings_overall_control": "CON.1",
            "pitching_ratings_overall_movement": "MOV",
            "pitching_ratings_overall_hra": "HRR",
            "pitching_ratings_overall_pbabip": "PBABIP",
            "pitching_ratings_vsr_stuff": "STU vR",
            "pitching_ratings_vsl_stuff": "STU vL",
            "pitching_ratings_vsr_control": "CON.1 vR",
            "pitching_ratings_vsl_control": "CON.1 vL",
            "pitching_ratings_vsr_movement": "MOV vR",
            "pitching_ratings_vsl_movement": "MOV vL",
            "pitching_ratings_vsr_hra": "HRR vR",
            "pitching_ratings_vsl_hra": "HRR vL",
            "pitching_ratings_vsr_pbabip": "PBABIP vR",
            "pitching_ratings_vsl_pbabip": "PBABIP vL",
            "pitching_ratings_talent_stuff": "STU P",
            "pitching_ratings_talent_control": "CON.1 P",
            "pitching_ratings_talent_movement": "MOV P",
            "pitching_ratings_talent_hra": "HRR P",
            "pitching_ratings_talent_pbabip": "PBABIP P",
            "pitching_ratings_pitches_fastball": "FB",
            "pitching_ratings_pitches_curveball": "CB",
            "pitching_ratings_pitches_slider": "SL",
            "pitching_ratings_pitches_changeup": "CH",
            "pitching_ratings_pitches_sinker": "SI",
            "pitching_ratings_pitches_cutter": "CT",
            "pitching_ratings_pitches_forkball": "FO",
            "pitching_ratings_pitches_splitter": "SP",
            "pitching_ratings_pitches_screwball": "SC",
            "pitching_ratings_pitches_circlechange": "CC",
            "pitching_ratings_pitches_knuckleball": "KN",
            "pitching_ratings_pitches_knucklecurve": "KC",
            "pitching_ratings_pitches_talent_fastball": "FBP",
            "pitching_ratings_pitches_talent_curveball": "CBP",
            "pitching_ratings_pitches_talent_slider": "SLP",
            "pitching_ratings_pitches_talent_changeup": "CHP",
            "pitching_ratings_pitches_talent_sinker": "SIP",
            "pitching_ratings_pitches_talent_cutter": "CTP",
            "pitching_ratings_pitches_talent_forkball": "FOP",
            "pitching_ratings_pitches_talent_splitter": "SPP",
            "pitching_ratings_pitches_talent_screwball": "SCP",
            "pitching_ratings_pitches_talent_circlechange": "CCP",
            "pitching_ratings_pitches_talent_knuckleball": "KNP",
            "pitching_ratings_pitches_talent_knucklecurve": "KCP",
            "pitching_ratings_misc_velocity": "VELO",
            "pitching_ratings_misc_velocity_target": "VT",
            "pitching_ratings_misc_arm_slot": "Slot",
            "pitching_ratings_misc_stamina": "STM",
            "pitching_ratings_misc_ground_fly": "G/F",
            "pitching_ratings_misc_hold": "HLD",
        },
        inplace=True,
    )

    # Calculate the number of pitches for each player
    pitch_columns = [
        "FB",
        "CB",
        "SL",
        "CH",
        "SI",
        "CT",
        "FO",
        "SP",
        "SC",
        "CC",
        "KN",
        "KC",
    ]
    df_player_pitching_ratings_base["PIT"] = (
        df_player_pitching_ratings_base[pitch_columns].gt(0).sum(axis=1)
    )

    df_player_ratings_base.set_index("ID", inplace=True)
    df_player_batting_ratings_base.set_index("ID", inplace=True)
    df_player_fielding_ratings_base.set_index("ID", inplace=True)
    df_player_pitching_ratings_base.set_index("ID", inplace=True)

    df_player_pitching_ratings_base.drop(
        columns=["league_id", "position", "role"], inplace=True
    )
    df_player_fielding_ratings_base.drop(
        columns=["league_id", "position", "role"], inplace=True
    )
    df_player_batting_ratings_base.drop(
        columns=["league_id", "position", "role"], inplace=True
    )

    df_player_ratings_base = df_player_ratings_base.merge(
        df_player_batting_ratings_base, on="ID", how="outer"
    )
    df_player_ratings_base = df_player_ratings_base.merge(
        df_player_fielding_ratings_base, on="ID", how="outer"
    )
    df_player_ratings_base = df_player_ratings_base.merge(
        df_player_pitching_ratings_base, on="ID", how="outer"
    )

    df_player_ratings_base["LPOS"] = df_player_ratings_base["LPOS"].apply(
        convert_dump_positions
    )

    df_player_ratings_base = df_player_ratings_base[df_player_ratings_base["ORG"] > 0]

    return df_player_ratings_base


BIZ_RENAMES = {
    "OPPS_0": "BIZ-R",
    "OPPS_1": "BIZ-L",
    "OPPS_2": "BIZ-E",
    "OPPS_3": "BIZ-U",
    "OPPS_4": "BIZ-Z",
    "OPPS_5": "BIZ-I",
    "OPPS_MADE_0": "BIZ-Rm",
    "OPPS_MADE_1": "BIZ-Lm",
    "OPPS_MADE_2": "BIZ-Em",
    "OPPS_MADE_3": "BIZ-Um",
    "OPPS_MADE_4": "BIZ-Zm",
    "OPPS_MADE_5": "BIZ-Im",
}

DUMP_TEAM_PITCHING_RENAMES = {"TEAM_ID": "Team Name", "S": "SV"}
DUMP_TEAM_BATTING_RENAMES = {
    "TEAM_ID": "Team Name",
    "K": "SO",
    "S": "SINGLE",
    "D": "DOUBLE",
    "T": "TRIPLE",
    "GDP": "GIDP",
}
DUMP_TEAM_FIELDING_RENAMES = {
    "TEAM_ID": "Team Name",
    "ER": "E",
    "POSITION": "POS",
    "FRAMING": "FRM",
    **BIZ_RENAMES,
}
DUMP_PLAYER_FIELDING_RENAMES = {
    "PLAYER_ID": "ID",
    "TEAM_ID": "ORG",
    "POSITION": "POS",
    "FRAMING": "FRM",
    **BIZ_RENAMES,
}
DUMP_PLAYER_PITCHING_RENAMES = {
    "PLAYER_ID": "ID",
    "TEAM_ID": "ORG",
    "SA": "SINGLE",
    "DA": "DOUBLE",
    "TA": "TRIPLE",
    "HRA": "HR",
    "IW": "IBB",
}
DUMP_PLAYER_BATTING_RENAMES = {
    "PLAYER_ID": "ID",
    "TEAM_ID": "ORG",
    "D": "DOUBLE",
    "T": "TRIPLE",
    "GDP": "GIDP",
    "K": "SO",
}

# split_id values in the career stat dumps
SPLIT_TOTAL, SPLIT_VS_LEFT, SPLIT_VS_RIGHT = 1, 2, 3
MAJOR_LEAGUE_LEVEL = 1

DUMP_BATTING_COLS = [
    "AB",
    "H",
    "SO",
    "PA",
    "G",
    "GS",
    "SINGLE",
    "DOUBLE",
    "TRIPLE",
    "HR",
    "R",
    "RBI",
    "HP",
    "SB",
    "CS",
    "BB",
    "IBB",
    "GIDP",
    "SH",
    "SF",
    "UBR",
]
DUMP_PITCHING_COLS = [
    "IP",
    "K",
    "BF",
    "BB",
    "R",
    "ER",
    "G",
    "GS",
    "W",
    "L",
    "S",
    "SF",
    "SH",
    "HR",
    "IBB",
    "DP",
    "HA",
    "AB",
    "SINGLE",
    "DOUBLE",
    "TRIPLE",
    "HP",
    "SB",
    "CS",
]
DUMP_FIELDING_COLS = [
    "TC",
    "A",
    "PO",
    "E",
    "IP",
    "G",
    "GS",
    "DP",
    "TP",
    "PB",
    "SBA",
    "RTO",
    "PLAYS",
    "ARM",
    "FRM",
] + list(BIZ_RENAMES.values())


def _team_innings(df: pd.DataFrame) -> pd.DataFrame:
    """Fold IPF (extra outs) into IP using baseball notation."""
    return df.assign(IP=outs_to_ip(dump_ip_to_outs(df))).drop(columns=["IPF"])


def _season_split(df: pd.DataFrame, season: int, split_id: int | None) -> pd.DataFrame:
    mask = (df["YEAR"] == season) & (df["LEVEL_ID"] == MAJOR_LEAGUE_LEVEL)
    if split_id is not None:
        mask &= df["SPLIT_ID"] == split_id
    return df.loc[mask]


def _batting_split(df: pd.DataFrame, season: int, split_id: int) -> pd.DataFrame:
    df = _season_split(df, season, split_id).rename(columns=DUMP_PLAYER_BATTING_RENAMES)
    df = df.assign(SINGLE=df["H"] - df["DOUBLE"] - df["TRIPLE"] - df["HR"])
    return df.groupby(["ID", "SPLIT_ID"])[DUMP_BATTING_COLS].sum().reset_index()


def _pitching_split(df: pd.DataFrame, season: int, split_id: int) -> pd.DataFrame:
    df = _season_split(df, season, split_id).rename(
        columns=DUMP_PLAYER_PITCHING_RENAMES
    )
    df = df.assign(IP=dump_ip_to_outs(df))
    grouped = df.groupby("ID")[DUMP_PITCHING_COLS].sum().reset_index()
    return grouped.assign(IP=outs_to_ip(grouped["IP"]))


def import_ootp_dump_stats(
    league: str, season: int, ratings_type: str, basedir: str
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:

    filedir = paths.dump_dir(league, ratings_type, season)

    def read(name: str) -> pd.DataFrame:
        df = pd.read_csv(f"{filedir}/{name}.csv")
        df.columns = df.columns.str.upper()
        return df

    df_team_pitching = _team_innings(read("team_pitching_stats")).rename(
        columns=DUMP_TEAM_PITCHING_RENAMES
    )
    df_team_batting = read("team_batting_stats").rename(
        columns=DUMP_TEAM_BATTING_RENAMES
    )
    df_team_fielding = _team_innings(read("team_fielding_stats_stats")).rename(
        columns=DUMP_TEAM_FIELDING_RENAMES
    )

    df_player_fielding = read("players_career_fielding_stats")
    df_player_batting = read("players_career_batting_stats")
    df_player_pitching = read("players_career_pitching_stats")

    df_player_fielding = _season_split(df_player_fielding, season, None).rename(
        columns=DUMP_PLAYER_FIELDING_RENAMES
    )
    df_player_fielding = df_player_fielding.assign(
        IP=dump_ip_to_outs(df_player_fielding)
    )
    df_player_fielding_grouped = (
        df_player_fielding.groupby(["ID", "POS"])[DUMP_FIELDING_COLS]
        .sum()
        .reset_index()
    )
    df_player_fielding_grouped["IP"] = outs_to_ip(df_player_fielding_grouped["IP"])

    return (
        _batting_split(df_player_batting, season, SPLIT_TOTAL),
        _pitching_split(df_player_pitching, season, SPLIT_TOTAL),
        df_player_fielding_grouped,
        df_team_batting,
        df_team_pitching,
        df_team_fielding,
        _batting_split(df_player_batting, season, SPLIT_VS_RIGHT),
        _batting_split(df_player_batting, season, SPLIT_VS_LEFT),
        _pitching_split(df_player_pitching, season, SPLIT_VS_RIGHT),
        _pitching_split(df_player_pitching, season, SPLIT_VS_LEFT),
    )


attribute_columns = [
    "LEA",
    "LOY",
    "FIN",
    "WE",
    "INT",
    # "PRONE",
    "CON",
    "GAP",
    "POW",
    "EYE",
    "K's",
    "BABIP",
    "CON vR",
    "GAP vR",
    "POW vR",
    "EYE vR",
    "K vR",
    "BA vR",
    "CON vL",
    "GAP vL",
    "POW vL",
    "EYE vL",
    "K vL",
    "BA vL",
    "CON P",
    "GAP P",
    "POW P",
    "EYE P",
    "K P",
    "BA P",
    "STU",
    "CON.1",
    "MOV",
    "HRR",
    "PBABIP",
    "STU vR",
    "STU vL",
    "CON.1 vR",
    "CON.1 vL",
    "MOV vR",
    "MOV vL",
    "HRR vR",
    "HRR vL",
    "PBABIP vR",
    "PBABIP vL",
    "STU P",
    "CON.1 P",
    "MOV P",
    "HRR P",
    "PBABIP P",
    "LF",
    "RF",
    "CF",
    "3B",
    "2B",
    "1B",
    "SS",
    "C",
    "P",
]

# text columns where "-" is a legitimate value
clean_skip = ["COMP", "HSC"]


def clean_ratings(df_player_ratings: pd.DataFrame) -> pd.DataFrame:

    set1 = set(df_player_ratings.columns)
    set2 = set(clean_skip)
    clean_columns = list(set1 - set2)
    df_player_ratings[clean_columns] = df_player_ratings[clean_columns].replace(
        "-", 0, regex=False
    )

    # Rating columns: "-" (not rated at that position/skill) and anything non-numeric become 0
    for col in attribute_columns:
        if col in df_player_ratings.columns:
            df_player_ratings[col] = (
                pd.to_numeric(df_player_ratings[col], errors="coerce")
                .fillna(0)
                .astype(int)
            )

    # FIXME
    # df_player_ratings["COMP"] = df_player_ratings["COMP"].replace(0, "None", regex=True)
    # df_player_ratings["HSC"] = df_player_ratings["COMP"].replace(0, "Noob", regex=True)
    # df_player_ratings["SLR"] = df_player_ratings["SLR"].replace(0, "$0", regex=True)
    # df_player_ratings["CV"] = df_player_ratings["CV"].replace(0, "$0", regex=True)

    # df_player_ratings = df_player_ratings.astype({"CV": str})
    return df_player_ratings


def import_files(
    league: str, season: int, source: str | None = None, ratings: str | None = None
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    save = get_save(league)
    source = source or save.source
    ratings = ratings or save.ratings

    if source == "dump":
        df_player_ratings = import_ootp_dump_ratings(league, season, ratings, ".")
        (
            df_player_batting_stats,
            df_player_pitching_stats,
            df_player_fielding_stats,
            df_team_batting_stats,
            df_team_pitching_stats,
            df_team_fielding_stats,
            df_player_batting_right_stats,
            df_player_batting_left_stats,
            df_player_pitching_right_stats,
            df_player_pitching_left_stats,
        ) = import_ootp_dump_stats(league, season, ratings, ".")
    else:
        (
            df_player_batting_stats,
            df_player_pitching_stats,
            df_player_fielding_stats,
            df_team_batting_stats,
            df_team_pitching_stats,
            df_team_fielding_stats,
        ) = import_csv_flex(league, season, ".")
        df_player_ratings = import_ratings(league, season, ratings, ".")
        df_player_batting_right_stats = pd.DataFrame()
        df_player_batting_left_stats = pd.DataFrame()
        df_player_pitching_right_stats = pd.DataFrame()
        df_player_pitching_left_stats = pd.DataFrame()

    df_player_ratings = clean_ratings(df_player_ratings)

    # check of export direction exists, if not create it
    paths.output_dir(league, season, create=True)

    df_player_ratings.to_csv(paths.output_file(league, season, "player-data"))

    df_team_batting_stats.round(3).to_csv(
        paths.output_file(league, season, "team-batting")
    )
    df_team_pitching_stats.round(3).to_csv(
        paths.output_file(league, season, "team-pitching")
    )
    df_team_fielding_stats.round(3).to_csv(
        paths.output_file(league, season, "team-fielding")
    )

    return (
        df_player_ratings,
        df_player_batting_stats,
        df_player_batting_right_stats,
        df_player_batting_left_stats,
        df_player_pitching_stats,
        df_player_pitching_right_stats,
        df_player_pitching_left_stats,
        df_player_fielding_stats,
        df_team_batting_stats,
        df_team_pitching_stats,
        df_team_fielding_stats,
    )

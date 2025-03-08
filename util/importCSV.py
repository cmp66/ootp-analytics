import pandas as pd
import os
from model import (
    convert_height_to_cm,
    convert_groundball_flyball,
    convert_velocity,
    convert_throws,
    convert_pitch_type,
    convert_slot,
    convert_bat_rl,
    convert_gbt,
    convert_fbt,
)


def import_csv_flex(
    league: str, season: int, basedir
) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:

    filedir = f"{basedir}/files/{league}/{season}"
    fielding_index = 0
    batting_index = 0
    pitching_index = 0
    fielding_dfs = [None] * 27
    batting_dfs = [None] * 10
    pitching_dfs = [None] * 10

    for file in [
        f for f in os.listdir(filedir) if os.path.isfile(os.path.join(filedir, f))
    ]:
        df_temp = pd.read_csv(f"{filedir}/{file}")

        # check if the PI/PA column exists
        if "CERA" in df_temp.columns:
            df_team_fielding = df_temp
        elif "PPG" in df_temp.columns:
            df_team_pitching = df_temp
        elif "wRC" in df_temp.columns:
            df_team_batting = df_temp
        elif "PI/PA" in df_temp.columns:
            batting_dfs[batting_index] = (
                batting_dfs[batting_index - 1].merge(df_temp, how="outer")
                if batting_index != 0
                else df_temp
            )
            batting_index += 1
        elif "IRS" in df_temp.columns:
            pitching_dfs[pitching_index] = (
                pitching_dfs[pitching_index - 1].merge(df_temp, how="outer")
                if pitching_index != 0
                else df_temp
            )
            pitching_index += 1
        elif "BIZ-R" in df_temp.columns:
            fielding_dfs[fielding_index] = (
                fielding_dfs[fielding_index - 1].merge(df_temp, how="outer")
                if fielding_index != 0
                else df_temp
            )
            fielding_index += 1

    df_player_fielding_stats = fielding_dfs[fielding_index - 1]
    df_player_batting_stats = batting_dfs[batting_index - 1]
    df_player_pitching_stats = pitching_dfs[pitching_index - 1]

    # Need to rename columns which have the same name in the two table (batting and ratings)
    df_player_batting_stats.rename(
        columns={"1B": "SINGLE", "2B": "DOUBLE", "3B": "TRIPLE", "BABIP": "BABIP-O"},
        inplace=True,
    )
    df_player_pitching_stats.rename(
        columns={
            "1B": "SINGLE",
            "2B": "DOUBLE",
            "3B": "TRIPLE",
            "BABIP": "BABIP-O",
            "FB": "FLYB",
            "GB": "GROUNDB",
        },
        inplace=True,
    )

    return (
        df_player_batting_stats,
        df_player_pitching_stats,
        df_player_fielding_stats,
        df_team_batting,
        df_team_pitching,
        df_team_fielding,
    )


def import_ratings(
    league: str, season: int, ratings_type: str, basedir: str
) -> pd.DataFrame:

    df_player_ratings = None

    filedir = f"{basedir}/files/{league}/{season}/ratings"
    print(os.listdir(filedir))
    print(f"Reading ratings from {filedir} of type {ratings_type}")
    for file in [
        f
        for f in os.listdir(filedir)
        if ratings_type in f and os.path.isfile(os.path.join(filedir, f))
    ]:
        print(f"Reading {file}")
        df_temp = pd.read_csv(f"{filedir}/{file}")
        df_player_ratings = (
            df_temp
            if df_player_ratings is None
            else df_player_ratings.merge(df_temp, how="outer")
        )

    df_player_ratings.rename(columns={"R": "ROOKIE"}, inplace=True)
    df_player_ratings = df_player_ratings.drop(columns=["POS", "EXP"])
    df_player_ratings.set_index("ID", inplace=True)

    df_player_ratings["WT"] = df_player_ratings["WT"].apply(lambda x: int(x[:3]))
    df_player_ratings["HT"] = df_player_ratings["HT"].apply(convert_height_to_cm)

    df_player_ratings["G/F"] = df_player_ratings["G/F"].apply(
        convert_groundball_flyball
    )
    df_player_ratings["VELO"] = df_player_ratings["VELO"].apply(convert_velocity)
    df_player_ratings["T"] = df_player_ratings["T"].apply(convert_throws)
    df_player_ratings["PT"] = df_player_ratings["PT"].apply(convert_pitch_type)
    df_player_ratings["Slot"] = df_player_ratings["Slot"].apply(convert_slot)

    # player_data["BBT"] = player_data["BBT"].apply(convert_bbt)
    df_player_ratings["GBT"] = df_player_ratings["GBT"].apply(convert_gbt)
    df_player_ratings["FBT"] = df_player_ratings["FBT"].apply(convert_fbt)
    df_player_ratings["B"] = df_player_ratings["B"].apply(convert_bat_rl)

    return df_player_ratings


def import_ootp_dump_ratings(
    league: str, season: int, ratings_type: str, basedir: str
) -> pd.DataFrame:
    filedir = f"{basedir}/files/{league}/dumps/{ratings_type}/{season}"

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
            "weight": "WT",
            "height": "HT",
            "bats": "B",
            "throws": "T",
            "age": "Age",
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
        columns=["league_id", "position", "role"], inplace=True, axis=1
    )
    df_player_fielding_ratings_base.drop(
        columns=["league_id", "position", "role"], inplace=True, axis=1
    )
    df_player_batting_ratings_base.drop(
        columns=["league_id", "position", "role"], inplace=True, axis=1
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

    df_player_ratings_base = df_player_ratings_base[df_player_ratings_base["ORG"] > 0]

    return df_player_ratings_base


def import_ootp_dump_stats(
    league: str, season: int, ratings_type: str, basedir: str
) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:

    filedir = f"{basedir}/files/{league}/dumps/{ratings_type}/{season}"

    df_team_pitching = pd.read_csv(f"{filedir}/team_pitching_stats.csv")
    df_team_batting = pd.read_csv(f"{filedir}/team_batting_stats.csv")
    df_team_fielding = pd.read_csv(f"{filedir}/team_fielding_stats_stats.csv")
    df_player_fielding = pd.read_csv(f"{filedir}/players_career_fielding_stats.csv")
    df_player_batting = pd.read_csv(f"{filedir}/players_career_batting_stats.csv")
    df_player_pitching = pd.read_csv(f"{filedir}/players_career_pitching_stats.csv")

    df_team_pitching.columns = df_team_pitching.columns.str.upper()
    df_team_batting.columns = df_team_batting.columns.str.upper()
    df_team_fielding.columns = df_team_fielding.columns.str.upper()
    df_player_fielding.columns = df_player_fielding.columns.str.upper()
    df_player_batting.columns = df_player_batting.columns.str.upper()
    df_player_pitching.columns = df_player_pitching.columns.str.upper()

    df_player_fielding = df_player_fielding[df_player_fielding["YEAR"] == season]
    df_player_batting = df_player_batting[
        (df_player_batting["YEAR"] == season) & (df_player_batting["SPLIT_ID"] == 1)
    ]
    df_player_pitching = df_player_pitching[
        (df_player_pitching["YEAR"] == season) & (df_player_pitching["SPLIT_ID"] == 1)
    ]

    df_team_pitching.rename(columns={"TEAM_ID": "Team Name", "S": "SV"}, inplace=True)

    batting_cols = [
        "AB",
        "H",
        "SO",
        "PA",
        "G",
        "GS",
        "D",
        "T",
        "HR",
        "R",
        "RBI",
        "SB",
        "CS",
        " BB",
        "IBB",
        "GIDP",
        "SH",
        "SF",
        "H",
        "UBR",
    ]

    df_team_batting.rename(
        columns={
            "TEAM_ID": "Team Name",
            "K": "SO",
            "S": "SINGLE",
            "D": "DOUBLE",
            "T": "TRIPLE",
            "GDP": "GIDP",
        },
        inplace=True,
    )

    df_team_fielding.rename(
        columns={
            "TEAM_ID": "Team Name",
            "ER": "E",
            "POSITION": "POS",
            "FRAMING": "FRM",
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
        },
        inplace=True,
    )

    fielding_cols = [
        "TC",
        "A",
        "PO",
        "E",
        "IP",
        "G",
        "GS",
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
    ]
    fielding_cols += [
        "BIZ-R",
        "BIZ-L",
        "BIZ-E",
        "BIZ-U",
        "BIZ-Z",
        "BIZ-I",
        "BIZ-Rm",
        "BIZ-Lm",
        "BIZ-Em",
        "BIZ-Um",
        "BIZ-Zm",
        "BIZ-Im",
    ]
    df_player_fielding.rename(
        columns={
            "PLAYER_ID": "ID",
            "TEAM_ID": "ORG",
            "POSITION": "POS",
            "FRAMING": "FRM",
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
        },
        inplace=True,
    )

    pitching_cols = [
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
    df_player_pitching.rename(
        columns={
            "PLAYER_ID": "ID",
            "TEAM_ID": "ORG",
            "SA": "SINGLE",
            "DA": "DOUBLE",
            "TA": "TRIPLE",
            "HRA": "HR",
            "IW": "IBB",
            # "ER": "E",
        },
        inplace=True,
    )

    # print(df_player_batting.columns)

    batting_cols = [
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
    df_player_batting.rename(
        columns={
            "PLAYER_ID": "ID",
            "TEAM_ID": "ORG",
            "D": "DOUBLE",
            "T": "TRIPLE",
            "GDP": "GIDP",
            "K": "SO",
        },
        inplace=True,
    )

    df_player_batting["SINGLE"] = (
        df_player_batting["H"]
        - df_player_batting["DOUBLE"]
        - df_player_batting["TRIPLE"]
        - df_player_batting["HR"]
    )

    df_player_fielding.set_index("ID", inplace=True)
    df_player_batting.set_index("ID", inplace=True)
    df_player_pitching.set_index("ID", inplace=True)

    df_player_batting_grouped = (
        df_player_batting.groupby("ID")[batting_cols].sum().reset_index()
    )
    df_player_pitching_grouped = (
        df_player_pitching.groupby("ID")[pitching_cols].sum().reset_index()
    )
    df_player_fielding_grouped = (
        df_player_fielding.groupby(["ID", "POS"])[fielding_cols].sum().reset_index()
    )

    # df_player_fielding_grouped.set_index("ID", inplace=True)
    # df_player_batting_grouped.set_index("ID", inplace=True)
    # df_player_pitching_grouped.set_index("ID", inplace=True)

    # print(df_player_batting_grouped.columns)
    # print(df_player_fielding_grouped.columns)
    # print(df_player_pitching_grouped.columns)

    return (
        df_player_batting_grouped,
        df_player_pitching_grouped,
        df_player_fielding_grouped,
        df_team_batting,
        df_team_pitching,
        df_team_fielding,
    )

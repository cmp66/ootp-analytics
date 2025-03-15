import pandas as pd
import os
import openpyxl


REPORTS_PATH = "./files/reports/"

RATINGS = "Player Ratings"
PLAYER_BATTING = "Player Batting"
PLAYER_PITCHING = "Player Pitching"
LEAGUE_BATTING = "League Batting"
LEAGUE_PITCHING = "League Pitching"
PLAYER_FIELDING = "Player Fielding"
LEAGUE_FIELDING = "League Fielding"
WOBA = "wOBA calcs"
FIELDING_PLAYABLES = "Fielding Playables"
PREDICTIONS = "Predictions"

TABS = [
    RATINGS,
    PLAYER_BATTING,
    PLAYER_PITCHING,
    LEAGUE_BATTING,
    LEAGUE_PITCHING,
    PLAYER_FIELDING,
    LEAGUE_FIELDING,
    WOBA,
    FIELDING_PLAYABLES,
    PREDICTIONS,
]


class ExcelReportWriter:

    def __init__(self, league: str):
        self.league = league
        self.filename = f"{REPORTS_PATH}/{league}.xlsx"

    def clear_workbook(self, sheet_name: str):

        wb = openpyxl.load_workbook(filename=self.filename)
        for ws in wb.worksheets:
            if ws.title == sheet_name:
                for row in ws.iter_rows(
                    min_row=2, max_row=ws.max_row, min_col=1, max_col=ws.max_column
                ):
                    for cell in row:
                        cell.value = None

        wb.save(self.filename)

    def load_player_stats(self, season: int):

        file_exists = os.path.exists(self.filename)

        if file_exists:
            for tab in TABS:
                self.clear_workbook(tab)

        ratings = pd.read_csv(
            f"./files/{self.league}/{season}/output/{self.league}-{season}-player-data.csv"
        )
        batting_stats = pd.read_csv(
            f"./files/{self.league}/{season}/output/{self.league}-{season}-hitting.csv"
        )
        pitching_stats = pd.read_csv(
            f"./files/{self.league}/{season}/output/{self.league}-{season}-pitching.csv"
        )
        fielding_stats = pd.read_csv(
            f"./files/{self.league}/{season}/output/{self.league}-{season}-fielding.csv"
        )
        league_batting_stats = pd.read_csv(
            f"./files/{self.league}/{season}/output/{self.league}-{season}-team-batting.csv"
        )
        league_pitching_stats = pd.read_csv(
            f"./files/{self.league}/{season}/output/{self.league}-{season}-team-pitching.csv"
        )
        league_fielding_stats = pd.read_csv(
            f"./files/{self.league}/{season}/output/{self.league}-{season}-team-fielding.csv"
        )
        woba = pd.read_csv(
            f"./files/{self.league}/{season}/output/{self.league}-{season}-woba-calcs.csv"
        )
        fielding_playables = pd.read_csv(
            f"./files/{self.league}/{season}/output/{self.league}-{season}-fielding-attributes.csv"
        )

        args = {
            "path": self.filename,
            "engine": "openpyxl",
            "mode": "a" if file_exists else "w",
        }

        args = args | ({"if_sheet_exists": "overlay"} if file_exists else {})

        with pd.ExcelWriter(**args) as writer:

            ratings.to_excel(writer, sheet_name=RATINGS, index=False)
            batting_stats.to_excel(writer, sheet_name=PLAYER_BATTING, index=False)
            pitching_stats.to_excel(writer, sheet_name=PLAYER_PITCHING, index=False)
            fielding_stats.to_excel(writer, sheet_name=PLAYER_FIELDING, index=False)
            league_batting_stats.to_excel(
                writer, sheet_name=LEAGUE_BATTING, index=False
            )
            league_pitching_stats.to_excel(
                writer, sheet_name=LEAGUE_PITCHING, index=False
            )
            league_fielding_stats.to_excel(
                writer, sheet_name=LEAGUE_FIELDING, index=False
            )
            woba.to_excel(writer, sheet_name=WOBA, index=False)
            fielding_playables.to_excel(
                writer, sheet_name=FIELDING_PLAYABLES, index=False
            )
            # Styler.to_excel(writer, sheet_name=RATINGS, float_format="%.3f")

        wb = openpyxl.load_workbook(filename=self.filename)
        if "Ratings" not in wb[RATINGS].tables:
            tab = openpyxl.worksheet.table.Table(
                displayName="Ratings",
                ref=f"A1:{openpyxl.utils.get_column_letter(ratings.shape[1])}{len(ratings)+1}",
            )
            wb[RATINGS].add_table(tab)
            wb.save(self.filename)

        if "PlayerBatting" not in wb[PLAYER_BATTING].tables:
            tab = openpyxl.worksheet.table.Table(
                displayName="PlayerBatting",
                ref=f"A1:{openpyxl.utils.get_column_letter(batting_stats.shape[1])}{len(batting_stats)+1}",
            )
            wb[PLAYER_BATTING].add_table(tab)
            wb.save(self.filename)

        if "PlayerPitching" not in wb[PLAYER_PITCHING].tables:
            tab = openpyxl.worksheet.table.Table(
                displayName="PlayerPitching",
                ref=f"A1:{openpyxl.utils.get_column_letter(pitching_stats.shape[1])}{len(pitching_stats)+1}",
            )
            wb[PLAYER_PITCHING].add_table(tab)
            wb.save(self.filename)

        if "PlayerFielding" not in wb[PLAYER_FIELDING].tables:
            tab = openpyxl.worksheet.table.Table(
                displayName="PlayerFielding",
                ref=f"A1:{openpyxl.utils.get_column_letter(fielding_stats.shape[1])}{len(fielding_stats)+1}",
            )
            wb[PLAYER_FIELDING].add_table(tab)
            wb.save(self.filename)

        if "LeagueFielding" not in wb[LEAGUE_FIELDING].tables:
            tab = openpyxl.worksheet.table.Table(
                displayName="LeagueFielding",
                ref=f"A1:{openpyxl.utils.get_column_letter(league_fielding_stats.shape[1])}{len(league_fielding_stats)+1}",
            )
            wb[LEAGUE_FIELDING].add_table(tab)
            wb.save(self.filename)

        if "LeagueBatting" not in wb[LEAGUE_BATTING].tables:
            tab = openpyxl.worksheet.table.Table(
                displayName="LeagueBatting",
                ref=f"A1:{openpyxl.utils.get_column_letter(league_batting_stats.shape[1])}{len(league_batting_stats)+1}",
            )
            wb[LEAGUE_BATTING].add_table(tab)
            wb.save(self.filename)

        if "LeaguePitching" not in wb[LEAGUE_PITCHING].tables:
            tab = openpyxl.worksheet.table.Table(
                displayName="LeaguePitching",
                ref=f"A1:{openpyxl.utils.get_column_letter(league_pitching_stats.shape[1])}{len(league_pitching_stats)+1}",
            )
            wb[LEAGUE_PITCHING].add_table(tab)
            wb.save(self.filename)

        if "wOBAcalcs" not in wb[WOBA].tables:
            tab = openpyxl.worksheet.table.Table(
                displayName="wOBAcalcs",
                ref=f"A1:{openpyxl.utils.get_column_letter(woba.shape[1])}{len(woba)+1}",
            )
            wb[WOBA].add_table(tab)
            wb.save(self.filename)

        if "FieldingPlayables" not in wb[FIELDING_PLAYABLES].tables:
            tab = openpyxl.worksheet.table.Table(
                displayName="FieldingPlayables",
                ref=f"A1:{openpyxl.utils.get_column_letter(fielding_playables.shape[1])}{len(fielding_playables)+1}",
            )
            wb[FIELDING_PLAYABLES].add_table(tab)
            wb.save(self.filename)

        wb.close()

    def write_predictions(self, df_predictions: pd.DataFrame):

        args = {
            "path": self.filename,
            "engine": "openpyxl",
            "mode": "a",
            "if_sheet_exists": "overlay",
        }

        with pd.ExcelWriter(**args) as writer:
            df_predictions.to_excel(writer, sheet_name=PREDICTIONS, index=False)

        wb = openpyxl.load_workbook(filename=self.filename)
        if PREDICTIONS not in wb[PREDICTIONS].tables:
            tab = openpyxl.worksheet.table.Table(
                displayName=PREDICTIONS,
                ref=f"A1:{openpyxl.utils.get_column_letter(df_predictions.shape[1])}{len(df_predictions)+1}",
            )
            wb[PREDICTIONS].add_table(tab)
            wb.save(self.filename)

        wb.close()

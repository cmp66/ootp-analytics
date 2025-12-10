# Description: This file contains the configuration for the reports module.
import requests
from io import StringIO
import pandas as pd

STATS_PLUS_LEAGUE_NAMES = {
    "Dugout": "thedugout",
    "TGS": "tgs",
}

STATS_BASE = "https://statsplus.net/"


def get_drafted_players_from_statsplus(league: str) -> list:

    url = f"{STATS_BASE}{STATS_PLUS_LEAGUE_NAMES[league]}/api/draft"
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch data from {url}")
    csv_dataas = StringIO(response.text)
    drafted_players = pd.read_csv(csv_dataas)
    drafted_list = drafted_players["ID"].tolist()

    return drafted_list
